-- Every AI result the system produces, in one columnar store.
--
-- What lives here: EVENTS — things that happened at a timestamp and are never
-- edited afterwards. That is what ClickHouse is for.
-- What stays in sqlite (siglip/reid.db): the matcher's mutable STATE — the
-- per-identity vector bank and the same-camera exclusivity claims are upserted
-- on every frame, which in a columnar store means rewriting parts.
--
-- ts is milliseconds since the epoch, as the pipeline already emits it.

CREATE DATABASE IF NOT EXISTS ccvt;

-- ---------------------------------------------------------------- detections
-- ~11.7 M rows/day today. ORDER BY (camera, ts) matches the only hot query:
-- "every box on this camera between two times", for the playback overlay.
CREATE TABLE IF NOT EXISTS ccvt.detections
(
    ts          DateTime64(3),
    camera      LowCardinality(String),
    class       LowCardinality(String),
    conf        Float32,
    track       Int64,                    -- -1 when the tracker gave none
    global_id   Int64,                    -- cross-camera identity, 0 = unassigned
    x           Float32,                  -- PIXELS in the (fw, fh) frame, not 0..1
    y           Float32,
    w           Float32,
    h           Float32,
    fw          UInt16,                   -- frame the box was measured in
    fh          UInt16
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(ts)
ORDER BY (camera, ts)
TTL toDateTime(ts) + INTERVAL 7 DAY
SETTINGS index_granularity = 8192;

-- --------------------------------------------------------------------- poses
-- 17 COCO keypoints per person (~64% of person records carry one). The pipeline
-- has always computed these on GPU 1 and the old bridge dropped them on the
-- floor. Three parallel arrays so a query can read only the joint it needs.
-- Same pixel space as detections: (fw, fh).
CREATE TABLE IF NOT EXISTS ccvt.poses
(
    ts          DateTime64(3),
    camera      LowCardinality(String),
    track       Int64,
    kp_x        Array(Float32),           -- length 17, PIXELS
    kp_y        Array(Float32),
    kp_conf     Array(Float32),           -- per-joint confidence
    pose_conf   Float32,                  -- the skeleton's own confidence
    fw          UInt16,
    fh          UInt16
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(ts)
ORDER BY (camera, ts)
TTL toDateTime(ts) + INTERVAL 3 DAY
SETTINGS index_granularity = 8192;

-- ------------------------------------------------------------------ sightings
-- Cross-camera person tracking: one row each time a Global ID was recognised on
-- a camera. No TTL — this is the retrospective record ("who took it, where did
-- they go"). Written by siglip, mirrored here.
CREATE TABLE IF NOT EXISTS ccvt.sightings
(
    ts          DateTime64(3),
    gid         UInt64,                   -- Global ID
    camera      LowCardinality(String),
    track       Int64,                    -- the local tracker id on that camera
    matched     UInt8,                    -- 0 = this is where the identity began
    score       Float32,                  -- max cosine that won the match
    n_obs       UInt16,                   -- crops pooled into the tracklet
    prev_cam    LowCardinality(String)
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (gid, ts);

-- -------------------------------------------------------------------- persons
-- One row per Global ID per description. ReplacingMergeTree keeps the newest,
-- so re-tagging a person overwrites rather than duplicates.
CREATE TABLE IF NOT EXISTS ccvt.persons
(
    gid         UInt64,
    described_ts DateTime64(3),
    mode        LowCardinality(String),   -- 'colour' | 'ir'  (measured, not asked)
    visibility  LowCardinality(String),
    sex         LowCardinality(String),
    upper       String,
    lower       String,
    carry       LowCardinality(String),
    head        LowCardinality(String),
    description String
)
ENGINE = ReplacingMergeTree(described_ts)
ORDER BY gid;

-- ------------------------------------------------------------------- segments
-- SAM 3 semantic scan of a camera's static scene: how many instances of each
-- prompted keyword it found, and the colour the overlay draws them in. Not a
-- time series and NOT masks — `calib/seg_scan.py` writes counts, not polygons,
-- so nothing here pretends to store geometry it does not have.
-- ReplacingMergeTree: a re-scan of a camera overwrites its previous scan.
CREATE TABLE IF NOT EXISTS ccvt.segments
(
    scanned_ts  DateTime64(3),
    camera      LowCardinality(String),
    label       LowCardinality(String),   -- the SAM 3 keyword, e.g. 'door'
    instances   UInt16,
    colour_bgr  Array(UInt8)              -- overlay colour, 3 channels
)
ENGINE = ReplacingMergeTree(scanned_ts)
ORDER BY (camera, label);

-- --------------------------------------------------------- minute rollup (MV)
-- The dashboard asks "how many distinct objects per minute per camera per
-- class". Keeping the state aggregate means the answer is precomputed at insert
-- time instead of scanning a day of detections.
CREATE TABLE IF NOT EXISTS ccvt.minute_stats
(
    minute      DateTime,
    camera      LowCardinality(String),
    class       LowCardinality(String),
    objects     AggregateFunction(uniq, Int64),
    records     AggregateFunction(count)
)
ENGINE = AggregatingMergeTree
PARTITION BY toYYYYMM(minute)
ORDER BY (camera, class, minute);

CREATE MATERIALIZED VIEW IF NOT EXISTS ccvt.minute_stats_mv TO ccvt.minute_stats AS
SELECT
    toStartOfMinute(ts) AS minute,
    camera,
    class,
    uniqState(track)    AS objects,
    countState()        AS records
FROM ccvt.detections
GROUP BY minute, camera, class;

-- ------------------------------------------------------------------ behaviors
-- What a person was DOING, as judged by the VLM on a full sub-stream frame.
-- Written only when a rule fires (a crowd forms, someone stands still, a track
-- vanishes, the parcel room is due for a look), never per frame.
--
-- `track` is the local tracker id, exactly as `sightings` records it, so a
-- behaviour joins back to the cross-camera identity through it. `global_id` is
-- also stored when the matcher had already assigned one at that instant — it is
-- 0 when the person had not yet been matched.
CREATE TABLE IF NOT EXISTS ccvt.behaviors
(
    ts          DateTime64(3),
    camera      LowCardinality(String),
    track       Int64,                    -- local tracker id; -1 = frame-level
    global_id   Int64,                    -- 0 = not matched yet
    trigger     LowCardinality(String),   -- crowd|dwell|vanish|parcel_sweep|parcel_new
    n_persons   UInt16,                   -- people in the frame when it fired
    activity    String,                   -- the VLM's one-line answer
    -- The exact frame the VLM was shown, on disk. Without it a verdict cannot be
    -- audited by a human and a RAG answer has no evidence to cite.
    snapshot    String,                   -- path under /output/siglip/behavior
    suspicious  UInt8,                    -- the VLM's own yes/no
    model       LowCardinality(String),
    latency_ms  UInt32,                   -- VLM round-trip
    -- staleness of the frame the VLM saw vs the moment the rule fired: near 0 once
    -- behaviour reads the detection-synced frame the pipeline already cut, instead
    -- of an out-of-sync gateway grab (which left ~half of vanish frames empty).
    frame_lag_ms UInt32 DEFAULT 0,
    -- keyword retrieval over the activity text ("carrying", "parcel", "door")
    INDEX ix_activity activity TYPE tokenbf_v1(4096, 3, 0) GRANULARITY 4
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(ts)
ORDER BY (camera, ts)
TTL toDateTime(ts) + INTERVAL 30 DAY;

-- ------------------------------------------------------------------- episodes
-- One row per (Global ID, camera, local track): everything that person did
-- during one continuous appearance on one camera, summarised in a sentence by
-- the LLM.
--
-- Why the summary happens HERE and not when the frame was captured: the vision
-- model is shown the picture with no idea what the room is, precisely so that it
-- cannot invent one ("a hospital", "a parking lot" — both were hallucinated).
-- An episode, by construction, is a single camera, so the room is known without
-- ambiguity and can be handed to the LLM safely at summary time.
CREATE TABLE IF NOT EXISTS ccvt.episodes
(
    gid         UInt64,
    camera      LowCardinality(String),
    track       Int64,
    first_ts    DateTime64(3),
    last_ts     DateTime64(3),
    n_events    UInt16,
    place       String,                   -- the camera's confirmed label
    summary     String,                   -- the LLM's sentence
    model       LowCardinality(String),
    made_ts     DateTime64(3)
)
ENGINE = ReplacingMergeTree(made_ts)
ORDER BY (gid, camera, track);

-- ------------------------------------------------------------------- articles
-- The narrative for one Global ID, written by the LLM from that person's
-- local-track blocks. Regenerated only when their behaviour record changes and
-- then goes quiet — a person still walking around has not finished their story.
CREATE TABLE IF NOT EXISTS ccvt.articles
(
    gid         UInt64,
    made_ts     DateTime64(3),
    n_episodes  UInt16,
    n_events    UInt32,                   -- what the article was written from
    places      String,                   -- the rooms it names, comma separated
    article     String,
    model       LowCardinality(String)
)
ENGINE = ReplacingMergeTree(made_ts)
ORDER BY gid;

-- ---------------------------------------------------------------------- faces
-- Who a tracked person is, when the face gallery recognises them.
--
-- The face is read from the SAME main-stream person crop the ReID pipeline
-- already cuts and the person gate already approved — no second detector, no
-- second frame fetch. Recognition is multi-shot: several crops of one local
-- track vote, so a single blurred frame cannot name someone.
--
-- `track` is the local tracker id and `global_id` the cross-camera identity, so
-- a face joins the movement record and the behaviour log through either.
CREATE TABLE IF NOT EXISTS ccvt.faces
(
    ts          DateTime64(3),
    camera      LowCardinality(String),
    track       Int64,
    global_id   Int64,                    -- 0 when the matcher had not assigned one
    person_id   LowCardinality(String),   -- '' = seen but not in the gallery
    person_name LowCardinality(String),
    score       Float32,                  -- best cosine against the gallery
    confidence  Float32,                  -- blended score/ratio the module returns
    votes       UInt16,                   -- crops that agreed
    of_votes    UInt16,                   -- crops embedded
    runner_up   LowCardinality(String)
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (camera, ts);

-- --------------------------------------------------------------------- plates
-- Thai licence plates, read on the car-park entry camera only.
--
-- One row per VEHICLE, not per frame: the module buffers crops of a vehicle
-- across frames and returns the confidence-weighted majority read when the
-- vehicle leaves. A single OCR of a moving plate is noise.
CREATE TABLE IF NOT EXISTS ccvt.plates
(
    ts          DateTime64(3),            -- when the vehicle left the frame
    first_ts    DateTime64(3),
    camera      LowCardinality(String),
    track       Int64,                    -- the vehicle's tracker id
    vehicle     LowCardinality(String),   -- car | motorcycle | truck | bus
    plate       String,                   -- the voted string
    confidence  Float32,
    votes       UInt16,                   -- reads that agreed
    reads       UInt16,                   -- OCR attempts
    snapshot    String                    -- best plate crop, on disk
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (camera, ts);

-- --------------------------------------------------------------- face_vectors
-- One 512-d face embedding per LOCAL TRACK, when a face was visible at all.
--
-- Why keep it. A face does not change with clothing, lighting or camera angle —
-- the three things that break an appearance embedding across cameras. So when a
-- face IS visible it is the strongest evidence two tracks are one person, and it
-- can merge Global IDs that the body vector never linked.
--
-- Why it cannot replace the body vector: measured on this camera set, only about
-- 10% of accepted person tracks ever show a face MediaPipe can align. The other
-- 90% are backs of heads, distant figures and overhead views.
--
-- `vec` is the module's own L2-normalised output, averaged over the crops of the
-- track that produced a face, then renormalised. `best_score` is the best cosine
-- against the enrolled gallery — 0 when nobody is enrolled.
CREATE TABLE IF NOT EXISTS ccvt.face_vectors
(
    ts          DateTime64(3),
    camera      LowCardinality(String),
    track       Int64,
    global_id   Int64,
    n_faces     UInt16,                   -- crops that yielded an aligned face
    best_score  Float32,
    vec         Array(Float32)
)
ENGINE = ReplacingMergeTree(ts)
ORDER BY (camera, track);

-- ------------------------------------------------------------------- vehicles
-- One row per VEHICLE track: what it looked like, and its plate when the plate
-- was large enough to read.
--
-- Tracking is not only about people. A car that parks by the parcel room at 3 am
-- is the same kind of fact as a person who stands beside it, and it is described
-- the same way: the VLM is shown the crop the detector already cut, and answers in
-- fixed fields. It is never told the plate, so it cannot invent one.
--
-- At nvr1_ch03 a plate is ~34 px wide — about seven pixels per character — so
-- `plate` is usually empty. That is a fact about where the camera points, not
-- about the OCR: `plate_px` records how wide the plate actually was.
CREATE TABLE IF NOT EXISTS ccvt.vehicles
(
    ts          DateTime64(3),            -- when it left the frame
    first_ts    DateTime64(3),
    camera      LowCardinality(String),
    track       Int64,
    class       LowCardinality(String),   -- car | motorcycle | truck | bus
    frames      UInt32,
    colour      LowCardinality(String),
    body        LowCardinality(String),   -- sedan, pickup, scooter, van …
    markings    String,                   -- anything written or carried
    description String,                   -- the one-line summary shown on a card
    plate       String,                   -- '' when it could not be read
    plate_conf  Float32,
    plate_px    UInt16,                   -- width of the biggest plate box seen
    snapshot    String                    -- the vehicle crop, on disk
)
ENGINE = ReplacingMergeTree(ts)
ORDER BY (camera, track);

-- VLM preprocessing of recorded video: one narration per person-active minute
-- per camera, written by the vlmscan service (activity-gated — empty minutes
-- are never sent to the VLM). Searched by the investigator via run_sql.
CREATE TABLE IF NOT EXISTS ccvt.clip_captions (
    ts        DateTime64(3),                -- start of the captioned minute
    camera    LowCardinality(String),
    segment   String,                       -- recording filename the frame came from
    persons   UInt16,                       -- distinct person tracks in that minute
    caption   String,                       -- VLM narration of the scene
    model     LowCardinality(String),
    latency_ms UInt32
) ENGINE = MergeTree ORDER BY (camera, ts);

-- VLM narration of each person-active recorded minute (written by vlmscan).
-- caption='' rows are done-markers for minutes with no usable footage.
CREATE TABLE IF NOT EXISTS ccvt.clip_captions (
    ts        DateTime64(3),
    camera    LowCardinality(String),
    segment   String,
    persons   UInt16,
    caption   String,
    model     LowCardinality(String),
    latency_ms UInt32
) ENGINE = MergeTree ORDER BY (camera, ts);

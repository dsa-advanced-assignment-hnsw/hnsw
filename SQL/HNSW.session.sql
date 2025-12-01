DROP TABLE image_raw_dataset;

CREATE TABLE image_raw_dataset (
    ImageID             TEXT, 
    Subset              TEXT, 
    OriginalURL         TEXT, 
    OriginalLandingURL  TEXT, 
    License             TEXT,
    AuthorProfileURL    TEXT, 
    Author              TEXT, 
    Title               TEXT, 
    OriginalSize        TEXT,   
    OriginalMD5         TEXT,
    Thumbnail300KURL    TEXT, 
    Rotation            TEXT
);

COPY image_raw_dataset
FROM '/mnt/newvolume/DSA/hnsw/SQL/train-images-boxable-with-rotation.csv'
WITH (FORMAT csv, HEADER true, QUOTE '"');

SELECT * FROM image_raw_dataset
LIMIT 1
OFFSET 0;
DROP TABLE IF EXISTS raw_image;

CREATE TABLE raw_image (
    idx BIGSERIAL PRIMARY KEY,
    ImageID TEXT UNIQUE NOT NULL,
    Subset TEXT,
    OriginalURL TEXT,
    OriginalLandingURL TEXT,
    License TEXT,
    AuthorProfileURL TEXT,
    Author TEXT,
    Title TEXT,
    OriginalSize TEXT,
    OriginalMD5 TEXT,
    Thumbnail300KURL TEXT,
    Rotation FLOAT
);

COPY raw_image(ImageID, Subset, OriginalURL, OriginalLandingURL, License,
               AuthorProfileURL, Author, Title, OriginalSize, OriginalMD5,
               Thumbnail300KURL, Rotation)
FROM '/media/huynguyen/New Volume/hnsw/.cache/train-images-boxable-with-rotation.csv'
DELIMITER ',' CSV HEADER;


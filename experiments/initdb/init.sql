DROP DATABASE  IF EXISTS openimages; 

CREATE DATABASE openimages;
\c openimages

CREATE TABLE Images (
  ImageID CHAR(16),
  Subset VARCHAR,
  OriginalURL VARCHAR,
  OriginalLandingURL VARCHAR,
  License VARCHAR,
  AuthorProfileURL VARCHAR,
  Author VARCHAR,
  Title VARCHAR,
  OriginalSize BIGINT,
  OriginalMD5 VARCHAR,
  Thumbnail300KURL VARCHAR,
  Rotation REAL,
  PRIMARY KEY(ImageID)
);

CREATE TABLE Dict (
  LabelName VARCHAR,
  DisplayLabelName VARCHAR,
  PRIMARY KEY (LabelName)
);


CREATE TABLE Labels (
  ImageID CHAR(16), -- REFERENCES Images(ImageID), 
  Source VARCHAR,
  LabelName VARCHAR REFERENCES Dict(LabelName),
  Confidence REAL,
  PRIMARY KEY(ImageID, Source, LabelName)
);

CREATE TABLE TrainableLabelNames (
  LabelName VARCHAR,
  PRIMARY KEY (LabelName)
);
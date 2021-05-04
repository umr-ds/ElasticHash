\c openimages

\COPY TrainableLabelNames FROM '/oi_csv/oidv6-classes-trainable.txt' DELIMITER ',' CSV;

\COPY Images FROM '/oi_csv/test-images-with-rotation.csv' DELIMITER ',' CSV HEADER;
\COPY Images FROM '/oi_csv/validation-images-with-rotation.csv' DELIMITER ',' CSV HEADER;
\COPY Images FROM '/oi_csv/oidv6-train-images-with-labels-with-rotation.csv' DELIMITER ',' CSV HEADER;

\COPY Dict FROM '/oi_csv/oidv6-class-descriptions.csv' DELIMITER ',' CSV;

\COPY Labels FROM '/oi_csv/oidv6-train-annotations-human-imagelabels.csv' DELIMITER ',' CSV HEADER;
\COPY Labels FROM '/oi_csv/train-annotations-machine-imagelabels.csv' DELIMITER ',' CSV HEADER;

\COPY Labels FROM '/oi_csv/validation-annotations-machine-imagelabels.csv' DELIMITER ',' CSV HEADER;
\COPY Labels FROM '/oi_csv/validation-annotations-human-imagelabels.csv' DELIMITER ',' CSV HEADER;

\COPY Labels FROM '/oi_csv/test-annotations-machine-imagelabels.csv' DELIMITER ',' CSV HEADER;
\COPY Labels FROM '/oi_csv/test-annotations-human-imagelabels.csv' DELIMITER ',' CSV HEADER;

UPDATE Labels SET labelname=trim(labelname);
UPDATE TrainableLabelNames SET labelname=trim(labelname);
UPDATE Dict SET labelname=trim(labelname);

CREATE INDEX ON Images (subset);
CREATE INDEX ON Labels (imageid);
CREATE INDEX ON Labels (confidence,imageid);
CREATE INDEX ON Labels (confidence,imageid,source);

VACUUM ANALYZE;
VACUUM FULL;
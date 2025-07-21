DROP TABLE IF EXISTS documents;

DROP SEQUENCE IF EXISTS doc_ids;

CREATE SEQUENCE doc_ids START 1;

CREATE TABLE documents(
	id	        INTEGER DEFAULT nextval('doc_ids') PRIMARY KEY,
	link 	    VARCHAR NOT NULL,
	title 	    VARCHAR,
	content     BLOB NOT NULL,
    image_url   VARCHAR
);	

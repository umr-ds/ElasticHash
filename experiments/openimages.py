# encoding: latin-1
# from config import cfg
import psycopg2

psycopg2.extensions.register_type(psycopg2.extensions.UNICODE)
psycopg2.extensions.register_type(psycopg2.extensions.UNICODEARRAY)


class OpenImagesDB:
    '''
    This class provides database queries for preprocessed Open Images postgresql db
    '''

    def __init__(self, host, database, user, password, port=5433):
        self.conn = psycopg2.connect(database=database, user=user, host=host, password=password, port=port)
        self.cur = self.conn.cursor()

    def get_image_labels(self, imageid, confidence=0.9):
        self.cur.execute("""
        select labelname from labels where imageid=%s and confidence >= %s
        """, (imageid, confidence))
        return [r[0] for r in self.cur.fetchall()]

    def get_image_labels_ids(self, imageid, confidence):
        """
        Returns a list with label ids for an image.
        :param imageid: image with labels
        :return: list of label ids
        """
        self.cur.execute("""
            SELECT id-1  AS id FROM labels
            JOIN (SELECT (ROW_NUMBER() OVER (ORDER BY LABELNAME)) AS id, labelname FROM TrainableLabelnames) AS labelids
            ON labelids.labelname = labels.labelname
            WHERE imageid=%s
            -- Execution time : ~10 ms
        """, (imageid,))
        return [r[0] for r in self.cur.fetchall()]

    def get_image_labels_with_text(self, imageid):
        """
        Returns a list with label ids and text annotations for imageid.
        :param imageid: image with labels
        :return: list of (labelid, displaylabelname)
        """
        self.cur.execute("""
            SELECT DISTINCT id-1  AS id,  dict.displaylabelname FROM labels
            JOIN (SELECT (ROW_NUMBER() OVER (ORDER BY LABELNAME)) AS id, labelname FROM TrainableLabelnames) AS labelids
            ON labelids.labelname = labels.labelname
            JOIN dict ON labels.labelname=dict.labelname
            WHERE imageid=%s
            -- Execution time : ~15 ms
        """, (imageid,))
        return [(r[0], r[1]) for r in self.cur.fetchall()]

    def get_image(self, imageid):
        """
        Returns information for an imageid:
        imageid, subset, originalurl, originalsize, thumbnail300kurl, downloaded
        :param imageid:
        :return: (imageid: int, subset:{train, validation}, originalurl: str, originalsize: int,
                 thumbnail300kurl: str, downloaded: bool)
        """
        self.cur.execute("""
            SELECT imageid, subset, originalurl, originalsize, thumbnail300kurl, downloaded FROM images WHERE imageid=%s
            -- Execution time : ~0.1 ms
        """, (imageid,))
        return self.cur.fetchone()

    def get_label_list(self):
        """
        Returns all available labels, containing id (0..num_labels), labelname, and display labelname.
        :return: list of (id, labelname,  displaylabelname)
        """
        self.cur.execute("""
            SELECT labelids.id-1  AS id, TrainableLabelnames.labelname, dict.displaylabelname FROM TrainableLabelnames
            JOIN (SELECT (ROW_NUMBER() OVER (ORDER BY LABELNAME)) AS id, labelname FROM TrainableLabelnames) AS labelids
            ON labelids.labelname = TrainableLabelnames.labelname
            JOIN dict ON TrainableLabelnames.labelname=dict.labelname
            -- Execution time : ~20 ms
        """)
        return self.cur.fetchall()

    def get_num_labels(self):
        """
        Number of labels used for training.
        :return: number of training labels
        """
        self.cur.execute("""
        SELECT count(*) FROM TrainableLabelnames""")
        return self.cur.fetchone()[0]

    def close(self):
        self.cur.close()
        self.conn.close()

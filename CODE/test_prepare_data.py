import unittest

import pandas as pd

import prepare_data as pd_


class MaskTests(unittest.TestCase):
    def test_masks_email_phone_url(self):
        out = pd_.mask_pii("Mail a.b@x.com, call (480) 555-1234 or see https://x.io/me")
        self.assertEqual(out, "Mail [EMAIL], call [PHONE] or see [URL]")

    def test_plain_text_untouched(self):
        self.assertEqual(pd_.mask_pii("Python and SQL"), "Python and SQL")


class CleanTests(unittest.TestCase):
    def test_postings_rules(self):
        long = "x" * 60
        df = pd.DataFrame({"job_id": [1, 2, 2, 3, 4],
                           "description": [long, long, long, "short", None]})
        out = pd_.clean_postings(df)
        self.assertEqual(out["job_id"].tolist(), [1, 2])

    def test_resumes_filter_and_mask(self):
        df = pd.DataFrame({"ID": [1, 2, 3], "Category": ["HR", "CHEF", "FINANCE"],
                           "Resume_str": ["mail me@a.com", "cook", None]})
        out = pd_.clean_resumes(df)
        self.assertEqual(out["ID"].tolist(), [1, 3])
        self.assertEqual(out.iloc[0]["Resume_clean"], "mail [EMAIL]")
        self.assertEqual(out.iloc[1]["Resume_clean"], "")


if __name__ == "__main__":
    unittest.main()

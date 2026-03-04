import unittest

from src.collect.utils import decode_response

class DecodeTests(unittest.TestCase):
    def test_forces_gbk_on_bad_detect(self):
        # build a fake response object with gbk-encoded Chinese chars but a
        # misleading encoding attribute. This mimics the behaviour observed
        # after "upgrade" when apparent_encoding returned mac_greek.
        class FakeResp:
            def __init__(self, content, encoding):
                self.content = content
                self.encoding = encoding
                # requests.Response.text would attempt to decode using
                # .encoding, but we don't rely on it inside our util.
                try:
                    self.text = content.decode(encoding)
                except Exception:
                    self.text = ""

        chinese = "竞彩足球".encode("gbk")
        fake = FakeResp(chinese, "mac_greek")
        out = decode_response(fake)
        self.assertIn("竞彩足球", out)

    def test_fallback_to_text_when_decode_fails(self):
        class FakeResp:
            def __init__(self):
                self.content = b"notascii"
                self.encoding = "utf-8"
                self.text = "fallback"

        fake = FakeResp()
        # force decode failure by swapping method temporarily
        orig = fake.content
        fake.content = None
        out = decode_response(fake)
        self.assertEqual(out, "fallback")
        fake.content = orig


if __name__ == "__main__":
    unittest.main()

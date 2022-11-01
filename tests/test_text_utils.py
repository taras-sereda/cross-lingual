from unittest import TestCase

from utils import split_on_speaker_change


class Test(TestCase):
    def test_split_on_speaker_change(self):
        text = """
        {taras_sereda}
        November 1, nice sunny day in Kyiv - capital of brave and resilient people.
        """
        res = split_on_speaker_change(text)
        self.assertEquals(len(res), 1)
        self.assertEquals(res['taras_sereda'].strip(),
                          "November 1, nice sunny day in Kyiv - capital of brave and resilient people.")

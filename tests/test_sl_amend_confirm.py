"""Broker-confirmed SL amend parsing."""
import unittest

from app.services.ctrader_client import _parse_amend_execution_event
from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAExecutionEvent
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOAExecutionType


class TestSlAmendConfirm(unittest.TestCase):
    def _event(self, etype, err=""):
        ev = ProtoOAExecutionEvent()
        ev.ctidTraderAccountId = 47516246
        ev.executionType = etype
        if err:
            ev.errorCode = err
        return ev.SerializeToString()

    def test_rejected_is_failed(self):
        payload = self._event(ProtoOAExecutionType.ORDER_REJECTED, err="INVALID_SL")
        res = _parse_amend_execution_event(payload, 45288097)
        self.assertFalse(res["ok"])
        self.assertEqual(res["result"], "failed")

    def test_accepted_is_confirmed(self):
        payload = self._event(ProtoOAExecutionType.ORDER_ACCEPTED)
        res = _parse_amend_execution_event(payload, 45288097)
        self.assertTrue(res["ok"])
        self.assertEqual(res["result"], "confirmed")


if __name__ == "__main__":
    unittest.main()

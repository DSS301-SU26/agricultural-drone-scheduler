from app.api.compat import dashboard_slots
import json

class MockBT:
    def add_task(self, *args, **kwargs): pass

res = dashboard_slots(location='Dong Thap', drone_model='DJI_T30', background_tasks=MockBT())
for slot in res['slots'][:1]:
    print('Slot time:', slot['timestamp'])
    for drone, eval_data in slot['decision_engine']['drones_eval'].items():
        print('  ', drone, '->', eval_data['decision'], 'final:', eval_data['final_decision'], 'is_safe_to_fly:', eval_data['is_safe_to_fly'])

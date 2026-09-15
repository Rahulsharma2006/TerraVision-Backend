from datetime import datetime, timezone

def audit_event(action, actor='analyst', **details):
    return {'timestamp':datetime.now(timezone.utc).isoformat(),'action':action,'actor':actor,'details':details}

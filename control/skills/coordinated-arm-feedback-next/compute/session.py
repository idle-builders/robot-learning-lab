"""Same attached source/start/consumer owner; no background provider monitor."""
from common import ARM, load
if __name__ == '__main__':
    load(ARM + '/session.py', 'feedback_next_retained_managed_session').main()

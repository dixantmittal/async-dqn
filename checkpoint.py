import time
from datetime import datetime


def checkpoint(shared_model, args):
    try:
        while True:
            time.sleep(args.checkpoint_frequency * 60)

            # Save model
            now = datetime.now().strftime("%d_%m_%H_%M")
            shared_model.save('trained/checkpoint_{}.model'.format(now))

    except KeyboardInterrupt:
        print('exiting checkpoint')

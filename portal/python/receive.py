import eel


@eel.expose
def log(x):
    print(x)


action = None


@eel.expose
def register_action(x):
    global action
    action = x


def get_frontend_action():
    global action

    action = None
    while action is None:
        eel.sleep(0.1)

    return action

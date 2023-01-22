import eel
import jinja2
from pprint import pprint, pformat

from game.utils import generate_hands
from game.game import TrackedGameRound


last_button_clicked = None


@eel.expose
def button_clicked(x):
    global last_button_clicked
    last_button_clicked = x


templateLoader = jinja2.FileSystemLoader(searchpath="./portal")
templateEnv = jinja2.Environment(loader=templateLoader)


eel.init('portal')
eel.start('main.html', block=False)


game = TrackedGameRound(0, generate_hands())
observation, actions = game.observe()
obs = pformat(observation, width=150)
obs = obs.replace('\n', '<br>')

eel.update_html("aa", obs)

aa = ["a", "b", "c"]

template = templateEnv.get_template("buttons.html")
outputText = template.render(buttons=aa)
eel.update_html("bb", outputText)

eel.register_buttons(aa)


aa = ["a", "e", "d"]

template = templateEnv.get_template("buttons.html")
outputText = template.render(buttons=aa)
eel.update_html("bb", outputText)
eel.register_buttons(aa)

while True:
    eel.sleep(1)

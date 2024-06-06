import pyglet

COLOUR_NAMES = {
    'TRANSPARENT': (255, 255, 255, 0),
	'BLACK':  (000, 000, 000, 255),
	'WHITE':  (255, 255, 255, 255),
	'RED':    (255, 000, 000, 255),
	'GREEN':  (000, 255, 000, 255),
	'BLUE':   (000, 000 ,255, 255),
	'GREY':   (100, 100, 100, 255),
	'PINK':   (255, 175, 175, 255),
	'YELLOW': (255, 255, 000, 255),
	'ORANGE': (255, 175, 000, 255),
	'PURPLE': (200, 000, 175, 200),
	'BROWN':  (125, 125, 100, 255),
	'AQUA':   (100, 230, 255, 255),
	'DARK_GREEN': (000, 100, 000, 255),
	'LIGHT_GREEN':(150, 255, 150, 255),
	'LIGHT_BLUE': (150, 150, 255, 255),
	'LIGHT_GREY': (200, 200, 200, 255),
	'LIGHT_PINK': (255, 230, 230, 255)
}

class GameWindow(pyglet.window.Window):
	MIN_UPS = 5
	def __init__(self, **kwargs):
		kwargs['config'] = pyglet.gl.Config(double_buffer=True, sample_buffers=1, samples=8)

		super(GameWindow, self).__init__(**kwargs)

		self.fpsDisplay = pyglet.window.FPSDisplay(self)

		self.batches = {
			"main": pyglet.graphics.Batch(),
		}
		self.labels = {
			"bot's thinking timer":		pyglet.text.Label('', x=950, y=self.height-20, color=COLOUR_NAMES['WHITE']),
		}

		self.add_handlers()

	def updateLabel(self, label, text='---'):
		if label in self.labels:
			self.labels[label].text = text
		

	def add_handlers(self):
		@self.event
		#didn't test this... whoops
		def on_resize(cx, cy):

			from game import game

		@self.event
		def on_mouse_press(x, y, button, modifiers):

			from game import game
			game.inputMouse(x, y, button, modifiers)


		@self.event
		def on_key_press(symbol, modifiers):

			from game import game
			game.inputKeyboard(symbol, modifiers)

		@self.event
		def on_draw():
      
			self.clear()

			self.batches["main"].draw()

			self.fpsDisplay.draw()

			for label in self.labels.values():
				label.draw()

	def get_batch(self, batch_name="main"):
		return self.batches[batch_name]

settings = {
		'width': 1200,
		'height': 800,
		'vsync': True,
		'resizable': False,
		'caption': "Waning Crescent Chess",
	}
	
window = GameWindow(**settings)

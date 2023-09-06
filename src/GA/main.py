import os
os.environ["KIVY_NO_CONSOLELOG"] = "1"


from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout

from kivy.lang.builder import Builder
from LSTMGAMain import main_ga

Builder.load_file('main.kv')

class LSTMGAWidget(BoxLayout):
    def __init__(self, **var_args):
        super(LSTMGAWidget, self).__init__(**var_args)
        self.orientation = 'vertical'
    
    def run_training(self, population_size, n_processes, minimum_epochs):
        main_ga(int(population_size), int(n_processes), int(minimum_epochs))


class LSTMGAApp(App):
    def build(self):
        return LSTMGAWidget()


if __name__ == '__main__':
    try:
        app = LSTMGAApp()
        app.run()
    except Exception as e:
        print(e)
        input("Press enter.")
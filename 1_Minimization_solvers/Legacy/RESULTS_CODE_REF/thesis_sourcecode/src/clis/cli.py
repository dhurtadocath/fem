import sys
import typer
import signal
from rich import print as rprint
from PyInquirer import prompt, print_json, Separator
from utils.cli_utils import get_list_of_available_models, load_and_test_a_patch_model, load_a_model,test_patch_model_with_a_point
app = typer.Typer()


@app.command("hi")
def sample1_func():
    rprint("[red bold]Hi[/red bold] [yellow]World[yello]")

@app.command("hello")
def sample_func():
    rprint("[red bold]Hello[/red bold] [yellow]World[yello]")

@app.command("load-model")
def load_model():
    
    # signal.pause()
    questions = [
        {
            "type": "list",
            "name": "model",
            "message": "Select a model:",
            "choices": get_list_of_available_models(),
            "default": None,
        },
        {
            'type': 'input',
            'name': '3d-point',
            'message': 'input the coordinates of a 3d-point:',
        },
        {
            'type': 'input',
            'name': 'more_input',
            'message': 'do you want to test another point?(y/n)',
        }
    ]


    answer = prompt(questions=questions[:2])
    loaded_model = load_a_model(answer['model'])
    predicted_patch = test_patch_model_with_a_point(loaded_model, answer['3d-point'])
    rprint(f"You have chosen model {answer['model']}")
    rprint(f"input point: {(answer['3d-point'])}")
    rprint(f"predicted patch: {predicted_patch}")

    answer = prompt(questions=questions[2])
    while(answer['more_input'] in ["y", "Y", "yes", "Yes", "YES"]):
        answer = prompt(questions=questions[1])
        predicted_patch = test_patch_model_with_a_point(loaded_model, answer['3d-point'])
        rprint(f"input point: {(answer['3d-point'])}")
        rprint(f"predicted patch: {predicted_patch}")
        answer = prompt(questions=questions[2])
    exit(0)

# TODO
@app.command("train-patch-model")
def train_patch_model():

    pass

if __name__ == "__main__":
    try:
        app()    
    except KeyboardInterrupt:
        sys.exit(0)
from techlandscape import (
    candidates,
    antiseed,
    expansion,
    robustness,
    errors,
    assets,
    utils,
    io,
)
import typer

app = typer.Typer()

app.add_typer(candidates.app, name="candidates")
app.add_typer(antiseed.app, name="antiseed")
app.add_typer(expansion.app, name="expansion")
app.add_typer(robustness.app, name="robustness")
app.add_typer(errors.app, name="model.errors")
app.add_typer(assets.app, name="assets")
app.add_typer(utils.app, name="utils")
app.add_typer(io.app, name="io")

if __name__ == "__main__":
    app()

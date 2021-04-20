import plotly.express as px
import plotly.offline as po

from techlandscape.config import Config
from techlandscape.decorators import monitor
from techlandscape.inference import stylized_facts
from techlandscape.inference.plots import fig_wrapper
from techlandscape.inference.utils import densify_var

# TODO? data transfo directly in the get_function
# TODO document
# TODO? transfer to utils? Not yet at least
# TODO make it executable

mapbox_token = Config().mapbox_token


def prepare_country_date_df(client, table_ref, data_path):
    df = stylized_facts.get_patent_country_date(client, table_ref, data_path)
    df["publication_year"] = df["publication_date"].apply(lambda x: int(x / 10000))
    return df


def prepare_entity_df(flavor, client, table_ref, data_path, country_date_df):
    assert flavor in ["inventor", "assignee"]
    df = stylized_facts.get_patent_entity(flavor, client, table_ref, data_path)
    df = df.merge(country_date_df, how="left", on="publication_number")
    df = df.merge(
        1 / df.groupby("publication_number").count().max(1).rename(f"nb_{flavor}"),
        how="left",
        on="publication_number",
    )
    return df


def prepare_geoc_df(client, table_ref, data_path, inventor_df, assignee_df):
    inventor_geoc_df = stylized_facts.get_patent_geoloc(
        "inv", client, table_ref, data_path
    )
    applicant_geoc_df = stylized_facts.get_patent_geoloc(
        "app", client, table_ref, data_path
    )
    inventor_geoc_df["type"] = "inventor"
    applicant_geoc_df["type"] = "applicant"
    inv_geoc_df = inventor_geoc_df.merge(
        inventor_df, on="publication_number", how="left"
    )
    app_geoc_df = applicant_geoc_df.merge(
        assignee_df, on="publication_number", how="left"
    )
    inv_geoc_df.columns = list(
        map(
            lambda x: x.replace("inventor_", "").replace("_inventor", ""),
            inv_geoc_df.columns,
        )
    )
    app_geoc_df.columns = list(
        map(
            lambda x: x.replace("assignee_", "").replace("_assignee", ""),
            app_geoc_df.columns,
        )
    )
    df = inv_geoc_df.append(app_geoc_df)
    return df


@monitor
def country_date_analysis(country_date_df, table_name, plots_path):
    # # nb patents by country
    fig = fig_wrapper(country_date_df, mode="bar", groups=["country_code"], f=len)
    fig.write_image(f"{plots_path}{table_name}_nb_patents_country.png")

    # # nb patents by year
    fig, tmp = fig_wrapper(
        country_date_df.query("publication_year>0"),
        mode="bar",
        groups=["publication_year"],
        f=len,
    )
    fig.write_image(f"{plots_path}{table_name}_nb_patents_year.png")

    # # nb patents by year & expansion_level
    fig = fig_wrapper(
        country_date_df.query("publication_year>0"),
        "bar",
        ["publication_year", "expansion_level"],
        f=len,
    )
    fig.write_image(f"{plots_path}{table_name}_nb_patents_year_expansion.png")

    # # nb patents by year & country
    fig = fig_wrapper(
        country_date_df.query("publication_year>0"),
        "bar",
        ["publication_year", "country_code"],
        f=len,
    )
    fig.write_image(f"{plots_path}{table_name}_nb_patents_year_country.png")

    fig = fig_wrapper(
        country_date_df.query("publication_year>0"),
        "line",
        ["publication_year", "country_code"],
        f=len,
        normalize=True,
        norm_axis=1,
    )
    fig.write_image(f"{plots_path}{table_name}_nb_patents_year_country-norm.png")


@monitor
def entity_analysis(flavor, entity_df, table_name, plots_path):
    assert flavor in ["inventor", "assignee"]
    tmp = entity_df.copy()
    tmp = densify_var(tmp, f"{flavor}_country", sum, f"nb_{flavor}", 10)
    tmp = densify_var(tmp, f"{flavor}_name", sum, f"nb_{flavor}", 10)

    # # nb patents by entity orig country
    fig = fig_wrapper(
        tmp[[f"{flavor}_country", f"nb_{flavor}"]], "bar", [f"{flavor}_country"], f=sum
    )
    fig.write_image(f"{plots_path}{table_name}_nb_patents_{flavor}-orig-country.png")

    # # nb patents by entity orig country by year
    fig = fig_wrapper(
        tmp[["publication_year", f"{flavor}_country", f"nb_{flavor}"]],
        "bar",
        ["publication_year", f"{flavor}_country"],
        f=sum,
    )
    fig.write_image(
        f"{plots_path}{table_name}_nb_patents_{flavor}-orig-country_year.png"
    )

    # # nb patents by entity
    fig = fig_wrapper(
        tmp[[f"{flavor}_name", f"nb_{flavor}"]], "bar", [f"{flavor}_name"], f=sum
    )
    fig.write_image(f"{plots_path}{table_name}_nb_patents_{flavor}.png")

    # # nb patent per entity per year
    fig = fig_wrapper(
        tmp[["publication_year", f"{flavor}_name", f"nb_{flavor}"]],
        "bar",
        ["publication_year", f"{flavor}_name"],
        f=sum,
    )
    fig.write_image(f"{plots_path}{table_name}_nb_patents_{flavor}_country_year.png")


@monitor
def geoc_analysis(geoc_df, table_name, plots_path):
    tmp = (
        geoc_df.groupby(["name", "type", "lat", "lng", "publication_year"])
        .apply(len)
        .rename("count")
        .reset_index()
    )
    # TODO? change to sum on nb -> nb contains NaNs
    tmp = tmp.sort_values("publication_year")
    px.set_mapbox_access_token(mapbox_token)
    fig = px.scatter_mapbox(
        tmp,
        lat="lat",
        lon="lng",
        color="type",
        hover_name="name",
        animation_frame="publication_year",
        size="count",
        size_max=15,
        zoom=1,
    )
    po.plot(
        fig, filename=f"{plots_path}{table_name}_all_geoc_year.html", auto_open=False
    )

    for e in ["inventor", "applicant"]:
        fig = px.scatter_mapbox(
            tmp.query("type==@e"),
            lat="lat",
            lon="lng",
            color="type",
            hover_name="name",
            animation_frame="publication_year",
            size="count",
            size_max=15,
            zoom=1,
        )
        po.plot(
            fig,
            filename=f"{plots_path}{table_name}_{e}_geoc_year.html",
            auto_open=False,
        )


def full_inference(table_name, data_path, plots_path, bq_config=None):

    config = bq_config if bq_config else Config()
    client = config.client()
    table_ref = config.table_ref(table_name, client)

    # expansion_df = expansion_analysis.count_expansion_level(client, table_ref, data_path)
    country_date_df = prepare_country_date_df(client, table_ref, data_path)
    inventor_df = prepare_entity_df(
        "inventor", client, table_ref, data_path, country_date_df
    )
    assignee_df = prepare_entity_df(
        "assignee", client, table_ref, data_path, country_date_df
    )
    geoc_df = prepare_geoc_df(client, table_ref, data_path, inventor_df, assignee_df)

    country_date_analysis(country_date_df, table_name, plots_path)
    entity_analysis("inventor", inventor_df, table_name, plots_path)
    entity_analysis("assignee", assignee_df, table_name, plots_path)
    geoc_analysis(geoc_df, table_name, plots_path)

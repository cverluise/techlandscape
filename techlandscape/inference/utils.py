def densify_var(df, group, f, var_name, n_largest):
    """
    Return <df> with a dense <group var>, meaning that only the <n_largest> (non null) categories
    wrt <var_name> are left unchanged. Other categories of <group> are changed into "others"
    :param df: pd.DataFrame
    :param group: list
    :param f: function
    :param var_name: str
    :param n_largest: int
    :return: pd.DataFrame
    """
    tmp = df.copy()
    var_clause = (
        tmp.groupby(group).apply(f)[var_name].nlargest(n_largest + 1).index
    )
    # filter out "" and None
    var_clause = list(filter(lambda x: x, var_clause))[:n_largest]
    dense_var = list(
        map(lambda x: x if x in var_clause else "others", tmp[group].values)
    )
    tmp[group] = dense_var
    return tmp

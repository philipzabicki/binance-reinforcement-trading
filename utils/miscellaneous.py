def convert_variables(variables):
    if isinstance(variables, dict):
        return variables
    elif isinstance(variables, list):
        return dict(variables)
    else:
        raise ValueError("Unsupported format for variables")

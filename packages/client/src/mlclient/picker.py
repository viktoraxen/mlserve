from InquirerPy import inquirer

from mlclient.model import Model


def pick(models: list[Model]) -> Model:
    columns = ["name", "created_at", "description"]
    entry_lists = [[] for _ in models]

    for column in columns:
        values = [str(getattr(m, column)) for m in models]
        width = max([len(e) for e in values])

        for i in range(len(entry_lists)):
            entry_lists[i].append(f"{values[i]:<{width}}")

    entries = []

    for entry_list in entry_lists:
        entries.append(" | ".join(entry_list))

    options = [{"name": e, "value": m} for e, m in zip(entries, models)]

    return inquirer.fuzzy(  # type: ignore[privateImportUsage]
        message="Select a model:",
        choices=options,
    ).execute()

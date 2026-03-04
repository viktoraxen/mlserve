from InquirerPy import inquirer


def pick(models: list[dict]) -> int:
    columns = ["name", "created_at", "description"]
    entry_lists = [[] for _ in models]

    for column in columns:
        values = [m[column] for m in models]
        width = max([len(e) for e in values])

        for i in range(len(entry_lists)):
            entry_lists[i].append(f"{values[i]:<{width}}")

    entries = []

    for entry_list in entry_lists:
        entries.append(" | ".join(entry_list))

    ids = [m["id"] for m in models]
    options = [{"name": e, "value": id} for e, id in zip(entries, ids)]

    return inquirer.fuzzy(  # type: ignore[privateImportUsage]
        message="Select a model:",
        choices=options,
    ).execute()

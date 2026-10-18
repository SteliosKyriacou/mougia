"""The two fly-brain ποιητάρηδες. Run labels in the browser and the paper's figures come from here."""

POETS = {
    "female": {"name": "Kyriakou", "connectome": "FlyWire FAFB v783 (female, brain)"},
    "male": {"name": "Markos", "connectome": "Male CNS v1.0 (male, brain portion)"},
}


def run_label(sex, wiring="connectome"):
    """e.g. run_label("female") -> "Kyriakou"; run_label("male", "random") -> "Markos (random wiring)"."""
    name = POETS[sex]["name"]
    return name if wiring == "connectome" else f"{name} ({wiring} wiring)"

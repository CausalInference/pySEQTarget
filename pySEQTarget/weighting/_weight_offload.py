def _offload_weights(self, boot_idx):
    """Offload fitted weight models to disk, replacing them with path refs.

    numerator_model/denominator_model are lists with one fit per treatment
    level; the cense/visit models are single fits. Entries already offloaded
    (str refs) or never fit (None) are left as-is. Consumers go through
    Offloader.load_model, which passes non-str values through.
    """
    for attr, name in (
        ("numerator_model", "numerator"),
        ("denominator_model", "denominator"),
    ):
        model_list = getattr(self, attr, None)
        if isinstance(model_list, list):
            for i, model in enumerate(model_list):
                if model is not None and not isinstance(model, str):
                    model_list[i] = self._offloader.save_model(
                        model, f"{name}{i}", boot_idx
                    )

    for attr, name in (
        ("cense_numerator_model", "cense_numerator"),
        ("cense_denominator_model", "cense_denominator"),
        ("visit_numerator_model", "visit_numerator"),
        ("visit_denominator_model", "visit_denominator"),
    ):
        model = getattr(self, attr, None)
        if model is not None and not isinstance(model, str):
            setattr(self, attr, self._offloader.save_model(model, name, boot_idx))

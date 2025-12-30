def _offload_weights(self, boot_idx):
    """Helper to offload weight models to disk"""
    weight_models = [
        ("numerator_model", "numerator"),
        ("denominator_model", "denominator"),
        ("LTFU_model", "LTFU"),
        ("visit_model", "visit"),
    ]

    for model_attr, model_name in weight_models:
        if hasattr(self, model_attr):
            model_list = getattr(self, model_attr)
            if model_list and isinstance(model_list, list) and len(model_list) > 0:
                latest_model = model_list[-1]
                if latest_model is not None:
                    offloaded = self._offloader.save_model(
                        latest_model, model_name, boot_idx
                    )
                    model_list[-1] = offloaded

"""A bounded rollout is complete only after its requested tick count."""
def complete(valid_steps,requested_steps,full_duration_complete=False,external_cutoff=True):
    if requested_steps:return valid_steps==requested_steps
    return bool(full_duration_complete and not external_cutoff)

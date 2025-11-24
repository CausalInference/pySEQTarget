def _format_time(start, end):
    elapsed = end - start
    days, rem = divmod(elapsed, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(days)}-{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"

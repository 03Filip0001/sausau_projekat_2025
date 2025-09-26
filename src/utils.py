def _print_msg(msg=None, end="#", sep=False, nl=False):
	print(msg)
	if sep:
		print(end*38); print()

	if nl:
		print()
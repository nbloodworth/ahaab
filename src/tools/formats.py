'''
AHAAB formats submodule
Part of the AHAAB tools module

ahaab/src
└──tools
    └──formats.py

Submodule list:

	notice
	warn
	error
'''

# Define some default values that can be called for quick custom formatting
colors={
	"RED":'\033[91m',
	"YELLOW":'\033[93m',
	"GREEN":'\033[92m',
	"BLUE":'\033[94m',
	"CYAN":'\033[96',
	"MAGENTA":'\033[95m',
	"BLACK":'\033[90m',
	"WHITE":'\033[97m'
}

formats={
	"END":'\033[0m',
	"BOLD":'\033[1m',
	"UNDERLINE":"\033[4m",
	"REGULAR":""
	}

cursor={
	"DELETELINE":"\x1b[2K"
}

# Define functions to quickly provide preset formatted messages
def message(msg,toprint=True,message_color="WHITE",message_format="REGULAR", end="\n"):
	'''
	message()

	Usage:
	$ message_msg=message(msg)

	Positional arguments:
	> msg: A string to display with formatting

	Keyword arguments:
	> toprint:       If True, print msg to stdout
	> message_color:  Color to use for message
	> message_format: Format ("BOLD" or 
	                 "UNDERLINE")

	Outputs:
	> String with specified ASCI formatting codes
	'''
	if message_color not in colors.keys():
		message_color="WHITE"
	if message_format not in formats.keys():
		message_format="REGULAR"

	fmsg='{}{}{}{}'.format(colors[message_color],formats[message_format],msg,formats["END"])
	if toprint:
		print(fmsg,end=end)
	else:
		return fmsg

def notice(msg,toprint=True,notice_color="WHITE",notice_format="BOLD", end="\n"):
	'''
	notice()

	Usage:
	$ notice_msg=notice(msg)

	Positional arguments:
	> msg: A string to display with formatting

	Keyword arguments:
	> toprint:       If True, print msg to stdout
	> notice_color:  Color to use for message
	> notice_format: Format ("BOLD" or 
	                 "UNDERLINE")

	Outputs:
	> String with specified ASCI formatting codes
	'''
	if notice_color not in colors.keys():
		notice_color="WHITE"
	if notice_format not in formats.keys():
		notice_format="BOLD"

	fmsg='{}{}{}{}'.format(colors[notice_color],formats[notice_format],msg,formats["END"])
	if toprint:
		print(fmsg, end=end)
	else:
		return fmsg

def warn(msg,toprint=True,warn_color="YELLOW",warn_format="BOLD", end="\n"):
	'''
	warn()

	Usage:
	$ warn_msg=warn(msg)

	Positional arguments:
	> msg: A string to display with formatting

	Keyword arguments:
	> toprint:       If True, print msg to stdout
	> warn_color:  Color to use for message
	> warn_format: Format ("BOLD" or 
	                 "UNDERLINE")

	Outputs:
	> String with specified ASCI formatting codes
	'''
	if warn_color not in colors.keys():
		warn_color="YELLOW"
	if warn_format not in formats.keys():
		warn_format="BOLD"

	fmsg='{}{}[WARNING]: {}{}'.format(colors[warn_color],formats[warn_format],msg,formats["END"])
	if toprint:
		print(fmsg, end=end)
	else:
		return fmsg

def error(msg,toprint=True,error_color="RED",error_format="BOLD", end="\n"):
	'''
	error()

	Usage:
	$ error_msg=error(msg)

	Positional arguments:
	> msg: A string to display with formatting

	Keyword arguments:
	> toprint:       If True, print msg to stdout
	> error_color:  Color to use for message
	> error_format: Format ("BOLD" or 
	                 "UNDERLINE")

	Outputs:
	> String with specified ASCI formatting codes
	'''
	if error_color not in colors.keys():
		error_color="RED"
	if error_format not in formats.keys():
		error_format="BOLD"

	fmsg='{}{}[ERROR]: {}{}'.format(colors[error_color],formats[error_format],msg,formats["END"])
	if toprint:
		print(fmsg, end=end)
	else:
		return fmsg
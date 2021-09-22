#!/bin/bash

# ------------------------------------------------------------------------------
# To run the script at a Linux or Mac OS X command-line:
#
#   $ bash nsidc-data-download.sh
#
# To run the script on Windows 10, you may try to use the
# Windows Subsystem for Linux, which allows you to run bash
# scripts from a command-line in Windows. See:
#
# https://blogs.windows.com/buildingapps/2016/03/30/run-bash-on-ubuntu-on-windows/
#
# To run the script on older versions of Windows, install a Unix-like command line
# utility such as Cygwin. After installing Cygwin (or a similar utility), change
# directories to the location of this script and run the command
# 'bash nsidc-data-download.sh' from the utility's command line.
#
# ------------------------------------------------------------------------------

includes_https=true

check_requirements() {
    if ! command -v curl >/dev/null 2>&1; then
        message="Could not find 'curl' program required for download. Please install curl before proceeding.

"
        printf "
Error: $message"; exit 1
    fi
}

get_credentials() {
    read -p "Earthdata userid: " userid
    read -s -p "Earthdata password: " password

    # Replace any existing .netrc entry with the latest userid/password
    netrc="machine urs.earthdata.nasa.gov"
    if [ -f ~/.netrc ]; then
        echo "Modifying your ~/.netrc file. Backup saved at ~/.netrc.bak"
        sed -i.bak "/$netrc/d" ~/.netrc
    fi
    echo "$netrc login $userid password $password" >> ~/.netrc
    chmod 0600 ~/.netrc
    unset password
}

authenticate() {
    printf "

Authenticating with Earthdata
"
    rm -f ~/.urs_cookies
    curl -s -b ~/.urs_cookies -c ~/.urs_cookies -L -n -o authn-data  'https://n5eil01u.ecs.nsidc.org/ICEBRIDGE/IODMS1B.001/2009.12.08/DMS_1000133_00333_20091208_22240330_V02.tif'
    if grep -q "Access denied" authn-data; then
        printf "
Error: could not authenticate to Earthdata. Please check your credentials and try again.

"
        rm authn-data
        exit 1
    fi
    rm authn-data
}

check_authorization() {
    printf "

Checking authorization with Earthdata
"
    result=`curl -# -H 'Origin: http://127.0.0.1:8080' -b ~/.urs_cookies  'https://urs.earthdata.nasa.gov/api/session/check_auth_status?client_id=_JLuwMHxb2xX6NwYTb4dRA'`
    echo $result
    if ! grep -q "true" <<<$result ; then
        printf "
Please ensure that you have authorized the NSIDC ECS DATAPOOL HTTPS ACCESS
Earthdata application in order to successfully download your data. This
only needs to be done once.

Please login to Earthdata by visiting the following link in your browser:

https://urs.earthdata.nasa.gov/home

And then authorize the Earthdata datapool application by visiting the
following link in your browser:

https://urs.earthdata.nasa.gov/approve_app?client_id=_JLuwMHxb2xX6NwYTb4dRA

"
        exit 1
    fi
}

fetch_urls() {
	echo $1
#-#, --progress-bar
#		Make curl display transfer progress as a simple progress bar  instead  of  the  standard,
#		more informational, meter.
#-O, --remote-name
#		Write  output  to  a local file named like the remote file we get. (Only the file part of
#		the remote file is used, the path is cut off.)
#-b, --cookie <data>
#		(HTTP) Pass the data to the HTTP server in the Cookie header. It is supposedly  the  data
#		previously  received  from the server in a Set-Cookie: line.  The data should be in the
#		format NAME1=VALUE1; NAME2=VALUE2.
#-c, --cookie-jar <filename>
#		(HTTP)  Specify to which file you want curl to write all cookies after a completed opera‐
#		tion. Curl writes all cookies from its in-memory cookie storage to the given file at  the
#		end  of  operations.  If  no cookies are known, no data will be written. The file will be
#		written using the Netscape cookie file format. If you set the file name to a single dash,
#		"-", the cookies will be written to stdout.
#-L, --location
#		(HTTP)  If  the  server reports that the requested page has moved to a different location
#		(indicated with a Location: header and a 3XX response code), this option will  make  curl
#		redo  the  request  on  the new place. If used together with -i, --include or -I, --head,
#		headers from all requested pages will be shown. When authentication is  used,  curl  only
#		sends  its credentials to the initial host. If a redirect takes curl to a different host,
#		it won't be able to intercept the user+password. See also --location-trusted  on  how  to
#		change  this.  You  can limit the amount of redirects to follow by using the --max-redirs
#		option.
#-n, --netrc
#		Makes  curl  scan  the  .netrc  (_netrc on Windows) file in the user's home directory for
#		login name and password. This is typically used for FTP on Unix. If used with HTTP,  curl
#		will enable user authentication. See netrc(5) ftp(1) for details on the file format. Curl
#		will not complain if that file doesn't have the  right  permissions  (it  should  not  be
#		either  world-  or  group-readable).  The environment variable HOME is used to find the
#		home directory.

    opts="-# -O -b ~/.urs_cookies -c ~/.urs_cookies -L -n"
    while read -r line; do
        retry=5
        status=1
        until [[ ( $status -eq 0 ) || ( $retry -eq 0 ) ]]; do
            echo "Downloading $line"
			curl $opts $line;
            status=$?
            retry=`expr $retry - 1`
        done
    done < $1
}

check_requirements

if [ "$includes_https" = true ]; then
    get_credentials
    authenticate
    check_authorization
fi

INPUT_FILE="./smap_download_links.txt"
fetch_urls $INPUT_FILE

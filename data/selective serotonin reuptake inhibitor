#!/bin/bash

# Public domain notice for all NCBI EDirect scripts is located at:
# https://www.ncbi.nlm.nih.gov/books/NBK179288/#chapter6.Public_Domain_Notice

# base="ftp://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect"
base="https://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect"

# function to fetch a single file, passed as an argument
FetchFile() {

  fl="$1"

  if [ -x "$(command -v curl)" ]
  then
    curl -s "${base}/${fl}" -o "${fl}"
  elif [ -x "$(command -v wget)" ]
  then
    wget "${base}/${fl}"
  else
    echo "Missing curl and wget commands, unable to download EDirect archive" >&2
    exit 1
  fi
}

# edirect folder to be installed in home directory
cd ~

# download and extract edirect archive
FetchFile "edirect.tar.gz"
if [ -s "edirect.tar.gz" ]
then
  gunzip -c edirect.tar.gz | tar xf -
  rm edirect.tar.gz
fi
if [ ! -d "edirect" ]
then
  echo "Unable to download EDirect archive" >&2
  exit 1
fi

# remaining executables to be installed within edirect folder
cd edirect

# get path for configuration file assistance commands
DIR=$( pwd )

plt=""
alt=""

# determine current computer platform
osname=$(uname -s)
cputype=$(uname -m)
case "$osname-$cputype" in
  Linux-x86_64 )
    plt=Linux
    ;;
  Darwin-x86_64 )
    plt=Darwin
    alt=Silicon
    ;;
  Darwin-*arm* )
    plt=Silicon
    alt=Darwin
    ;;
  CYGWIN_NT-* | MINGW*-* )
    plt=CYGWIN_NT
    ;;
  Linux-*arm* )
    plt=ARM
    ;;
  Linux-aarch64 )
    plt=ARM64
    ;;
  * )
    echo "Unrecognized platform: $osname-$cputype"
    exit 1
    ;;
esac

# fetch appropriate precompiled versions of xtract, rchive, and transmute
if [ -n "$plt" ]
then
  for exc in xtract rchive transmute
  do
    FetchFile "$exc.$plt.gz"
    gunzip -f "$exc.$plt.gz"
    chmod +x "$exc.$plt"
    if [ -n "$alt" ]
    then
      # for Apple Silicon, download both versions of executable
      FetchFile "$exc.$alt.gz"
      gunzip -f "$exc.$alt.gz"
      chmod +x "$exc.$alt"
    fi
  done
fi

# offer to edit the PATH variable assignment command in configuration files
echo ""
echo "Entrez Direct has been successfully downloaded and installed."
echo ""

prfx="In order to complete the configuration process, please execute the following:"
advice=$(mktemp)

target=bash_profile
if ! grep "$target" "${HOME}/.bashrc" >/dev/null 2>&1
then
  if [ ! -f ${HOME}/.$target ] || grep 'bashrc' "${HOME}/.$target" >/dev/null 2>&1
  then
    target=bashrc
  else
    if [ -n "$prfx" ]
    then
      printf "$prfx\n\n"
      prfx=""
    fi
    printf "  echo \"source ~/.bash_profile\" >> \${HOME}/.bashrc\n" | tee $advice
  fi
fi
if ! grep "PATH[^:]*edirect" "${HOME}/.$target" >/dev/null 2>&1
then
  if [ -n "$prfx" ]
  then
    printf "$prfx\n\n"
    prfx=""
  fi
  printf "  echo \"export PATH=${DIR}:\\\${PATH}\" >> \${HOME}/.$target\n" | tee -a $advice
fi
zarget=zshrc
if [ "$osname" = "Darwin" ]
then
  if [ ! -f ${HOME}/.zshrc ] && [ -f ${HOME}/.zprofile ] >/dev/null 2>&1
  then
    zarget=zprofile
  fi
  if ! grep "PATH[^:]*edirect" "${HOME}/.$zarget" >/dev/null 2>&1
  then
    if [ -n "$prfx" ]
    then
      printf "$prfx\n\n"
      prfx=""
    fi
    printf "  echo \"export PATH=${DIR}:\\\${PATH}\" >> \${HOME}/.$zarget\n" | tee -a $advice
  fi
fi

if [ -z "$prfx" ]
then
  echo ""
  if [ "$osname" = "Darwin" ]
  then
    echo "or manually edit the PATH variable assignments in your .$target and .$zarget files."
  else
    echo "or manually edit the PATH variable assignment in your .$target file."
  fi
  echo ""
  echo "Would you like to do that automatically now? [y/N]"
  read response
  case "$response" in
    [Yy]*      ) . $advice; echo "OK, done." ;;
    [Nn]* | '' ) echo "Holding off, then." ;;
    *          ) echo "Conservatively taking that as a no." ;;
  esac
fi
rm $advice

case ":$PATH:" in
  *:$HOME/edirect:*)
    ;;
  *)
    echo ""
    echo "To activate EDirect for this terminal session, please execute the following:"
    echo ""
    printf "export PATH=\${HOME}/edirect:\${PATH}\n"
    echo ""
    ;;
esac

for x in $target $zarget
do
  f=$HOME/.$x
  if grep -q "PATH.*:.*edirect" $f 2>/dev/null
  then
    echo
    echo "Please consider removing these obsolete lines from $f:"
    echo
    grep "PATH.*:.*edirect" $f
  fi
done

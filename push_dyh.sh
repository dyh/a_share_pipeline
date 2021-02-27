echo "#################### config global user name & email ####################"
git config --global user.email "1358366+dyh@users.noreply.github.com"
git config --global user.name "dyh"

echo "#################### git add . ####################"
git add .

echo "#################### git pull ####################"
git pull

echo "#################### git commit -m \"daily\" ####################"
git commit -m "tested ppo and a2c with cashpenalty env"

echo "#################### git push -u origin env ####################"
git push -u origin env

echo "#################### done ####################"

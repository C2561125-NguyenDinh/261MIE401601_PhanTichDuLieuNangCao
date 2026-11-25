@echo off
chcp 65001 >nul
REM Day toan bo ma nguon tieu luan len GitHub. Chay bang cach nhap dup vao tep nay.
cd /d "%~dp0"
where git >nul 2>nul || (echo Chua cai Git. Tai tai https://git-scm.com/download/win roi chay lai. & pause & exit /b 1)
if not exist .git git init
git config user.name >nul 2>nul || git config user.name "Dinh Nguyen Tan Nguyen"
git config user.email >nul 2>nul || git config user.email "C2561125-NguyenDinh@users.noreply.github.com"
git checkout -B main
git add -A
git commit -m "Tieu luan PTDLNC: he thong phan bo ngan sach khuyen mai theo uplift (X5 RetailHero)"
git remote remove origin 2>nul
git remote add origin https://github.com/C2561125-NguyenDinh/261MIE401601_PhanTichDuLieuNangCao.git
git push -u origin main
if errorlevel 1 (echo. & echo Day len that bai. Neu GitHub da co san tep README, chay: git pull origin main --allow-unrelated-histories roi chay lai tep nay.) else (echo. & echo Da day xong: https://github.com/C2561125-NguyenDinh/261MIE401601_PhanTichDuLieuNangCao)
pause

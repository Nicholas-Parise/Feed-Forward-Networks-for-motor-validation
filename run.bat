FOR /L %%i IN (1,1,10) DO (
python NN.py L30fft_64.out
timeout /t 1 /nobreak
)

FOR /L %%i IN (1,1,10) DO (
python NN.py L30fft150.out
timeout /t 2 /nobreak
)

FOR /L %%i IN (1,1,10) DO (
python NN.py L30fft1000.out
timeout /t 1 /nobreak
)

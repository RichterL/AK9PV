# Task 1 - Součet vektorů pomocí CUDA API.

> 1. Pomocí jazyka C/C++, CUDA API a vhodných vývojových nástrojů (CUDA Online IDE, NVIDIA NSight, CodeLite+SSH,...) vytvořte aplikaci implementující paralelní součet prvků dvou vektorů o velikosti n inicializovaných na hodnoty 1 - n. Hodnotu n volte v řádech stovek až tisíců.
> 2. Ověřte funkcionalitu na vzdáleném překladovém serveru a GPGPU Tesla K40.
> 3. Otestujte možnosti debuggingu CUDA kernelů, tj. vložení breakpointu do kernelu a jeho krokování (vyžaduje IDE NSight a dodatečnou grafickou kartu).
> 4. Otestujte možnosti profilingu CUDA kernelů, tj. spuštění vzdálené aplikace v prostředí NSight Visual Profiler (vyžaduje IDE NSight).

K debuggingu jsem využil konzolového nástroje cuda-gdb, umožňuje v kódu umístit breakpointy a následně průběh programu krokovat. IDE, které jsem použil bohužel nepodporuje CUDA debugging a ve VS Code se mi ho nepodařilo zprovoznit. Debugging přímo v IDE je podstatně užitečnější a umožňuje přístup k proměnným bez nutnosti je "pollovat" přes CLI.

Pro zajímavější výstup profilingu jsem musel zvolit ještě vyšší řád než bylo zadáno (10^9), aby se projevily dopady. Vektory jsem plnil přímo na GPU, při plnění na CPU se 80-90 % programu odehrávalo na CPU. Celkové využití RAM přesáhlo 11GB. Pro moji verzi GK zatím není možné využít sběru GPU metrik.
![nVidia Nsights](nsights.png)
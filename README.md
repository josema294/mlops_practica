# Practica de MLOps

## Master DL universidad Politecnica de Madrid 

## Jose Maria Aranguren Palma

---

Para la asignatura de MLOps, he decidido que usare como proyecto de referencia con el que realizar todas las buenas practicas, el TFM del master. De este modo podre aprovechar para crear dicho TFM con todas las buenas practicas aprendidas en esta asignatura y al mismo tiempo poder trasladar a futuro lo que vaya realizando aqui.

>Como resumen de dicho proyecto, se trata de un sistema de DL de deteccion de anomalias en infraestructura ferroviaria, obviamente al no poder viajar cientos de veces por toda la red de trenes del pais, para nuestro caso se ha optado por una aproximacion tipo conceptual o de simulacion de la viabilidad del proyecto en forma de maquieta de tren.

La mencionada maqueta llevara adosado una combinacion de microcontroller esp32+acelerometro, de modo que ira recogiendo datos en los ejes tridimensioanles del movimiento. Estos datos son informacion numerica que puede ser tratada y usada por un modelo de DL que sera capaz de discernir, cuando estos patrones son normales y entran dentro de lo que esperiamos en una via y un movimiento normal para la maqueta, o cuando hay anomalias que pueden producir problemas de confort e incluso situaciones de peligro.

No podemos obviamente tener el proyecto completo con todas las funcionalidades y capacidades para la entrega de esta asignatura por una cuestion meramente de tiempo, asi la version aqui vista sera un modelo altamente simplificado en un estado temprano de desarrollo. Esto implica que habra que hacer una serie de concesiones como no perder tiempo recolectando la enorme cantidad de datos que necesitamos y por contra usar datos sinteticos que intenten emular lo que nos encontrariamos de la manera mas aproximada. Por  otro lado el modelo se reducira a una version basica de Autoencoder sin muchas "florituras" pero minimanete funcional para lo que aqui necesitamos, ya que lo primordial es poner en uso las buenas practicas aprendias en la asignatura.


---


El proyecto se ha diseñado para ir cumpliendo los requerimientos de la asignatura, recogidos en este MD y que estan disponibles abajo. Esta lista a modo de checklist, se ira rellenando y complimentando a medida que avancemos con el proyecto, y ademas se ira commiteando todo ello de forma progresiva para sacar todo el partido al control de versiones con git.


- [x] **Creacion de carpeta y repositorio git:** Creado en el primer commit.
- [x] **Estructura de Proyecto:** Organizo el proyecto en las carpetas esenciales segun lo visto en clase (`src/`, `data/`, `models/`, `notebooks/`,`tests/`) y archivos accesorios esenciales, como el Dockerfile, .gitignore etc.

 
 
 
 
 
 

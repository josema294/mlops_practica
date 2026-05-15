# Practica de MLOps

## Master DL universidad Politecnica de Madrid 

## Jose Maria Aranguren Palma

---

Para la asignatura de MLOps, he decidido que usare como proyecto de referencia con el que realizar todas las buenas practicas, el TFM del master. De este modo podre aprovechar para crear dicho TFM con todas las buenas practicas aprendidas en esta asignatura y al mismo tiempo poder trasladar a futuro lo que vaya realizando aqui.

> Como resumen de dicho proyecto, se trata de un sistema de DL de deteccion de anomalias en infraestructura ferroviaria, obviamente al no poder viajar cientos de veces por toda la red de trenes del pais, para nuestro caso se ha optado por una aproximacion tipo conceptual o de simulacion de la viabilidad del proyecto en forma de maquieta de tren. La mencionada maqueta llevara adosado una combinacion de microcontroller esp32+acelerometro, de modo que ira recogiendo datos en los ejes tridimensioanles del movimiento. Estos datos son informacion numerica que puede ser tratada y usada por un modelo de DL que sera capaz de discernir, cuando estos patrones son normales y entran dentro de lo que esperiamos en una via y un movimiento normal para la maqueta, o cuando hay anomalias que pueden producir problemas de confort e incluso situaciones de peligro.

No podemos obviamente tener el proyecto completo con todas las funcionalidades y capacidades para la entrega de esta asignatura por una cuestion meramente de tiempo, asi la version aqui vista sera un modelo altamente simplificado en un estado temprano de desarrollo. Esto implica que habra que hacer una serie de concesiones como no perder tiempo recolectando la enorme cantidad de datos que necesitamos y por contra usar datos sinteticos que intenten emular lo que nos encontrariamos de la manera mas aproximada. Por  otro lado el modelo se reducira a una version basica de Autoencoder sin muchas "florituras" pero minimanete funcional para lo que aqui necesitamos, ya que lo primordial es poner en uso las buenas practicas aprendias en la asignatura.


---




## Instrucciones de uso

- Clonar el repositorio

- Construir dockerbuild:

```bash
docker build -t "tag_de_la_imagen" .
```

- Levantar el notebook:

```bash
docker run --rm -p 8888:8888 -v "$(pwd):/app" mlops_proyecto
```

- Correr entrenamiento:

```bash
docker run --rm -v "$(pwd):/app" mlops_proyecto python -m src.train
```

- Correr test:

```bash
docker run --rm -v "$(pwd):/app" mlops_proyecto python -m pytest tests
```

- Levantar API inferencia:

```bash
docker run --rm -p 8000:8000 -v "$(pwd):/app" mlops_proyecto uvicorn src.inference_api:app --host 0.0.0.0 --port 8000
```

## W&B

El proyecto de W&B esta publico en:

https://wandb.ai/jmaranguren89-upm/train-anomaly-detection?nw=nwuserjmaranguren89

Y el link del report es el siguiente:

https://wandb.ai/jmaranguren89-upm/train-anomaly-detection/reports/Analisis-anomalias-en-infraestructura-ferroviaria---VmlldzoxNjg4NDMzOA


El proyecto siguiendo los conceptos vistos en clase, no guarda el modelo .pth, este esta en w&b que he configurado con visibilidad publica. La api de inferencia esta configurado para usar el modelo de la carpeta models/ que no esta trackeado por git, y no existiendo intenta bajarselo de W&B, en local con .env con la api key ha funcionado sin problemas, por si acaso incluso siendo publico el proyecto no lo descargara, se puede reentrenar le modelo que tarda poco, o tambien lo he subido como release a github para poder ser descargado y metido en la carpeta models/.

link modelo:

https://github.com/josema294/mlops_practica/releases/tag/modelos

---


El proyecto se ha diseñado para ir cumpliendo los requerimientos de la asignatura, recogidos en este MD y que estan disponibles abajo. Esta lista a modo de checklist, se ira rellenando y complimentando a medida que avancemos con el proyecto, y ademas se ira commiteando todo ello de forma progresiva para sacar todo el partido al control de versiones con git.


- [x] **Creacion de carpeta y repositorio git:** Creado en el primer commit.
- [x] **Estructura de Proyecto:** Organizo el proyecto en las carpetas esenciales segun lo visto en clase (`src/`, `data/`, `models/`, `notebooks/`,`tests/`) y archivos accesorios esenciales, como el Dockerfile, .gitignore etc.
- [x] **Docker archivos configuracion:** Relleno el dockerfile con imagen pythonslim como visto en clase y el .dockerignore con una plantilla de ignores tipicos.
- [x] **Creacion del notebook para experimentaciones:** Creoacion del cuadernillo para pruebas y experimentacion con modelos.
- [x] **Notebook:** Paso el notebook con la base del proyecto de DL, autoencoder para deteccion de anomalias. 
- [x] **Pasamos noteebok a archivos python:** Tras haber creado el modelo funcional de pruenbas, empezamos a crear arvhivos mas profesionales en src/ incluyendo loggin y w&b cargado con apikey en .env
- [x] **Runs de pruebas, seleccion de modelo en W&B:** Se corren multiples modelos y pruebas y se elige modelo ganador, los detalles se especificaran en el reporte de W&B 
- [x] **Creada api de inferencia:**  Se crea la api, autocontenida con su propio html,css,js para poder despleghar tambien un frontend usable por el profesor/evaluador.
- [x] **Agrego enlace a reporte de W&B:** Se agrega en readme enalce a reporte de W&B, tambien version pdf subida al repositorio por su surgiera problemas de acceso al enlace.
- [X] **Datos de testeo:** Para poder usar la api de inferencia se necesitan datos de vibraciones de la via, como es logico que el evaluador del proyecto no los tenga, creo carpeta co ndatos ligeros en tests/datatest/, estos son los datos sinteticos creados para poder probar en la api de inferencia, uno pensado especificamente para ser similar a los datos sinteticos utilizados para el entrenamiento, y otro pensado para tener picos y alteraciones que hagan detectarlo como anomalia. Esctos csv se podran subir a la api para probarla cuando este levantada. Ademas se usan en los test para comprobar que el modelo responde como esta esperado.
- [X] **Inclusion de test:** Se incluyen test de pyhton y se actuyaliza el requirements.txt 
 
 

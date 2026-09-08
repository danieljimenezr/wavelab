# El pipeline, y por qué está así

Este fichero es el **piloto** para migrar DR Solutions. Las decisiones están tomadas pensando en
que se copie, no en que funcione una vez.

## Las cuatro decisiones que importan

**1. La clave de despliegue no es una llave del servidor.** En `authorized_keys` va con
`command="/root/cd_deploy.sh $SSH_ORIGINAL_COMMAND"` más `no-pty,no-port-forwarding,
no-agent-forwarding`. Aunque alguien se hiciera con ella, **solo puede ejecutar el despliegue**:
no abre una shell, no reenvía puertos, no lee un fichero. Es la diferencia entre una clave de
despliegue y acceso root, y no cuesta nada.

**2. La reversión automática vive en el SERVIDOR, no en el pipeline.** Si el runner de GitHub se
cae a mitad del despliegue, la reversión tiene que ocurrir igual. `cd_deploy.sh` comprueba la salud
tras reiniciar y, si falla, vuelve solo al SHA anterior y lo deja funcionando. Un pipeline que
despliega pero no revierte solo traslada el trabajo manual a la peor hora posible.

**3. `cancel-in-progress: false` en los despliegues.** Cancelar un despliegue a mitad deja el
servidor en un estado indeterminado — dependencias a medio instalar, servicio parado. Es peor que
hacer esperar al siguiente. En los *tests* sí conviene cancelar; en los despliegues, nunca.

**4. Los tests de red están excluidos.** Los marcados `net` dependen de que Binance responda. Un
pipeline que falla por una API ajena enseña al equipo a ignorar los fallos, y un pipeline que se
ignora es peor que no tener pipeline.

## Secretos

| secreto | qué es |
|---|---|
| `SSH_DEPLOY_KEY` | clave privada ed25519, restringida por `command=` |
| `SSH_KNOWN_HOSTS` | huella del host, para no desactivar la verificación |
| `VPS_HOST` | IP del servidor |

Nunca se usa `StrictHostKeyChecking=no`. Desactivar la verificación del host convierte cualquier
secuestro de DNS o de ruta en una entrega de la clave privada al atacante.

## Para replicarlo en otro repositorio

1. `bash scripts/cd_setup.sh` en el servidor destino (genera la clave y la restringe).
2. Volcar la privada al secreto sin que pase por pantalla:
   `ssh root@host 'cat /root/.ssh/deploy' | gh secret set SSH_DEPLOY_KEY --repo owner/repo`
3. Copiar `cd_deploy.sh` adaptando el nombre del servicio y la URL de salud.
4. Copiar este workflow.

El paso 2 importa: una clave que se imprime en una terminal, en un log o en un chat está
comprometida desde ese instante. Ocurrió durante el montaje de este mismo pipeline y hubo que
regenerarla.

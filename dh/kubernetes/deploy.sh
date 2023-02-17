namespace=dh
kubectl config set-context --current --namespace=${namespace}
kubectl apply -f dh-deploy.yaml

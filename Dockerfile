FROM golang:onbuild
EXPOSE 8080
RUN go build ./main.go
CMD ["./main"]

package main

import (
	"github.com/gorilla/websocket"
	"net/http"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:1024,
	WriteBufferSize:1024,
	CheckOrigin: func(r *http.Request) bool{
		return true
	},
}

type Client struct{
	hub *Hub
	conn *websocket.Conn
}


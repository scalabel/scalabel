package main

import (
	"log"

	pb "../proto"
	"google.golang.org/grpc"
)

type Hub struct {
	config            *Configuration
	registerSession   chan *Session
	unregisterSession chan *Session
	sessions          map[string]*Session
	grpcConnection    *grpc.ClientConn
	modelServer       pb.ModelServerClient
}

func newhub(config *Configuration) *Hub {
	grpcConnection, err := grpc.Dial(config.MachineHost+":"+config.MachinePort,
		grpc.WithInsecure())
	if err != nil {
		log.Fatalf("Fail to dial: %v", err)
	}
	modelServer := pb.NewModelServerClient(grpcConnection)

	return &Hub{
		config:            config,
		sessions:          make(map[string]*Session),
		registerSession:   make(chan *Session),
		unregisterSession: make(chan *Session),
		grpcConnection:    grpcConnection,
		modelServer:       modelServer,
	}
}

func (h *Hub) run() {
	defer h.grpcConnection.Close()
	for {
		select {
		case sess := <-h.registerSession:
			h.sessions[sess.uuid] = sess
		case sess := <-h.unregisterSession:
			sessID := sess.uuid
			_, ok := h.sessions[sessID]
			if ok {
				delete(h.sessions, sessID)
			}
		}

	}
}

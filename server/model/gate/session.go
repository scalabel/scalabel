package main

import (
	pb "../proto"
	"encoding/json"
	"fmt"
	"github.com/gorilla/websocket"
	"golang.org/x/net/context"
	"log"
	"time"
)

const (
	pongWait   = 1800 * time.Second
	pingPeriod = (pongWait * 9) / 10
)

type Session struct {
	uuid         string
	client       *Client
	appName      string
	modelName    string
	modelVersion int
}

type AppMessage struct {
	SessionId string `json:"sessionId"`
	StartTime string `json:"startTime"`
}

type SessionResponse struct {
	SessionId  string        `json:"sessionId"`
	TimingData DummyResponse `json:"timingData"`
}

type Message struct {
	Type    string          `json:"type"`
	Message json.RawMessage `json:"message"`
}

func startSession(hub *Hub, conn *websocket.Conn) {
	var msg AppMessage
	err := conn.ReadJSON(&msg)
	log.Printf("Got this message: %v at %s\n", msg, time.Now().String())

	if err != nil {
		log.Println("Register App ReadJSON Error", err)
		conn.Close()
	}

	var session *Session
	if existingSession, ok := hub.sessions[msg.SessionId]; ok {
		session = existingSession
	} else {
		echoedMessage, modelServerTimestamp, modelServerDuration, grpcDuration := hub.grpcRegistration(msg.SessionId)

		session = &Session{
			uuid: msg.SessionId,
			client: &Client{
				hub:  hub,
				conn: conn,
			},
		}

		hub.registerSession <- session
		timingData := DummyResponse{
			EchoedMessage:        echoedMessage,
			ModelServerTimestamp: modelServerTimestamp,
			ModelServerDuration:  modelServerDuration,
			GrpcDuration:         grpcDuration,
			StartTime:            msg.StartTime,
		}
		registrationResponse := SessionResponse{
			SessionId:  session.uuid,
			TimingData: timingData,
		}
		session.client.conn.WriteJSON(&registrationResponse)
		go session.DataListener()
	}
}

//Call the Register remote procedure and get the timing data
func (hub *Hub) grpcRegistration(sessionId string) (string, string, string, string) {
	ctx, cancel := context.WithTimeout(context.Background(), 10 * time.Second)
	defer cancel()

	start := time.Now()
	response, err := hub.modelServer.Register(ctx, &pb.Session{Message: "register", SessionId: sessionId})
	end := time.Now()
	grpcDuration := fmt.Sprintf("%.3f", float64(end.Sub(start))/float64(time.Millisecond))

	if err != nil {
		log.Fatalf("could not register with gRPC: %v", err)
	}
	return response.Session.Message, response.ModelServerTimestamp, response.ModelServerDuration, grpcDuration
}

type DummyData struct {
	Message   string `json:"message"`
	StartTime string `json:"startTime"`
	TerminateSession string `json:"terminateSession"`
}

type DummyResponse struct {
	EchoedMessage        string `json:"echoedMessage"`
	ModelServerTimestamp string `json:"modelServerTimestamp"`
	ModelServerDuration  string `json:"modelServerDuration"`
	GrpcDuration         string `json:"grpcDuration"`
	StartTime            string `json:"startTime"`
}

func (session *Session) DataListener() {
	defer func() {
		log.Println("Close DataListener.")
		session.client.conn.Close()
	}()

	for {
		var msg DummyData
		err := session.client.conn.ReadJSON(&msg)
		log.Printf("Got this message: %v at %s\n", msg, time.Now().String())
		if err != nil {
			log.Println("Invalid message")
			session.client.hub.unregisterSession <- session
			break
		}

		if msg.TerminateSession == "true" {
			log.Println("Terminating go session")
			session.grpcKill()
			session.client.hub.unregisterSession <- session
			break
		}

		echoedMessage, modelServerTimestamp, modelServerDuration, grpcDuration := session.grpcComputation(msg)

		dummyResponse := DummyResponse{
			EchoedMessage:        echoedMessage,
			ModelServerTimestamp: modelServerTimestamp,
			ModelServerDuration:  modelServerDuration,
			GrpcDuration:         grpcDuration,
			StartTime:            msg.StartTime,
		}
		session.client.conn.WriteJSON(&dummyResponse)
	}
}

//Call the DummyComputation remote procedure with DummyData, and get data for DummyResponse
func (session *Session) grpcComputation(msg DummyData) (string, string, string, string) {
	ctx, cancel := context.WithTimeout(context.Background(), 10 * time.Second)
	defer cancel()

	start := time.Now()
	response, err := session.client.hub.modelServer.DummyComputation(
		ctx, &pb.Session{Message: msg.Message, SessionId: session.uuid})
	end := time.Now()
	grpcDuration := fmt.Sprintf("%.3f", float64(end.Sub(start))/float64(time.Millisecond))
	if err != nil {
		log.Fatalf("could not echo from gRPC: %v", err)
	}
	return response.Session.Message, response.ModelServerTimestamp, response.ModelServerDuration, grpcDuration
}

//Kill the ray actor corresponding to the go session being killed
func (session *Session) grpcKill() {
	ctx, cancel := context.WithTimeout(context.Background(), 10 * time.Second)
	defer cancel()

	_, err := session.client.hub.modelServer.KillActor(
		ctx, &pb.Session{Message: "", SessionId: session.uuid})

	if err != nil {
		log.Fatalf("could not kill ray worker using grpc: %v", err)
	}
}

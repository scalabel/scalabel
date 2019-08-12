/* Credit to some open source resources for the authentication part:
   https://github.com/mura123yasu/go-cognito/blob/master/verifyToken.go
   https://gist.github.com/MathieuMailhos/361f24316d2de29e8d41e808e0071b13 */

package main

import (
	"crypto/rsa"
	"encoding/base64"
	b64 "encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"math/big"
	"net/http"
	"strings"
	"time"

	jwt "github.com/dgrijalva/jwt-go"
)

func validateAccessToken(tokenStr, region, userPoolId string,
	jwk map[string]JWKKey) (*jwt.Token, error) {
	// Decode the token string into JWT format.
	token, err := jwt.Parse(tokenStr, func(token *jwt.Token) (interface{},
		error) {

		// cognito user pool : RS256
		if _, ok := token.Method.(*jwt.SigningMethodRSA); !ok {
			return nil, fmt.Errorf("Unexpected signing method: %v",
				token.Header["alg"])
		}

		// 5. Get the kid from the JWT token header and
		// retrieve the corresponding JSON Web Key that was stored
		if kid, ok := token.Header["kid"]; ok {
			if kidStr, ok := kid.(string); ok {
				key := jwk[kidStr]
				// 6. Verify the signature of the decoded JWT token.
				rsaPublicKey := convertKey(key.E, key.N)
				return rsaPublicKey, nil
			}
		}

		// retrieved rsa public key
		return "", nil
	})

	if err != nil {
		return token, err
	}

	claims := token.Claims.(jwt.MapClaims)

	iss, ok := claims["iss"]
	if !ok {
		return token, fmt.Errorf("token does not contain issuer")
	}
	issStr := iss.(string)
	if strings.Contains(issStr, "cognito-idp") {
		// 3. 4. 7. Together with checking
		err = validateAWSJwtClaims(claims, region, userPoolId)
		if err != nil {
			return token, err
		}
	}

	if token.Valid {
		return token, nil
	}
	return token, err
}

// validateAWSJwtClaims validates AWS Cognito User Pool JWT
func validateAWSJwtClaims(claims jwt.MapClaims, region,
	userPoolId string) error {
	var err error
	// 3. Check the iss claim. It should match your user pool.
	issShoudBe := fmt.Sprintf("https://cognito-idp.%v.amazonaws.com/%v",
		region, userPoolId)
	err = validateClaimItem("iss", []string{issShoudBe}, claims)
	if err != nil {
		return err
	}

	// 4. Check the token_use claim.
	validateTokenUse := func() error {
		if tokenUse, ok := claims["token_use"]; ok {
			if tokenUseStr, ok := tokenUse.(string); ok {
				if tokenUseStr == "id" || tokenUseStr == "access" {
					return nil
				}
			}
		}
		return errors.New("token_use should be id or access")
	}

	err = validateTokenUse()
	if err != nil {
		return err
	}

	// 7. Check the exp claim and make sure the token is not expired.
	err = validateExpired(claims)
	if err != nil {
		return err
	}

	return nil
}

func validateClaimItem(key string, keyShouldBe []string,
	claims jwt.MapClaims) error {
	if val, ok := claims[key]; ok {
		if valStr, ok := val.(string); ok {
			for _, shouldbe := range keyShouldBe {
				if valStr == shouldbe {
					return nil
				}
			}
		}
	}
	return fmt.Errorf("%v does not match any of valid values: %v",
		key, keyShouldBe)
}

func validateExpired(claims jwt.MapClaims) error {
	if tokenExp, ok := claims["exp"]; ok {
		if exp, ok := tokenExp.(float64); ok {
			now := time.Now().Unix()
			//fmt.Printf("current unixtime : %v\n", now)
			//fmt.Printf("expire unixtime  : %v\n", int64(exp))
			if int64(exp) > now {
				return nil
			}
		}
		return errors.New("cannot parse token exp")
	}
	return errors.New("token is expired")
}

func convertKey(rawE, rawN string) *rsa.PublicKey {
	decodedE, err := base64.RawURLEncoding.DecodeString(rawE)
	if err != nil {
		panic(err)
	}
	if len(decodedE) < 4 {
		ndata := make([]byte, 4)
		copy(ndata[4-len(decodedE):], decodedE)
		decodedE = ndata
	}
	pubKey := &rsa.PublicKey{
		N: &big.Int{},
		E: int(binary.BigEndian.Uint32(decodedE[:])),
	}
	decodedN, err := base64.RawURLEncoding.DecodeString(rawN)
	if err != nil {
		panic(err)
	}
	pubKey.N.SetBytes(decodedN)
	// fmt.Println(decodedN)
	// fmt.Println(decodedE)
	// fmt.Printf("%#v\n", *pubKey)
	return pubKey
}

// JWK is json data struct for JSON Web Key
type JWK struct {
	Keys []JWKKey
}

// JWKKey is json data struct for cognito jwk key
type JWKKey struct {
	Alg string
	E   string
	Kid string
	Kty string
	N   string
	Use string
}

func getJWK(jwkUrl string) map[string]JWKKey {

	jwk := &JWK{}

	err := getJson(jwkUrl, jwk)
	if err != nil {
		Error.Println(err)
	}
	jwkMap := make(map[string]JWKKey)
	for _, jwk := range jwk.Keys {
		jwkMap[jwk.Kid] = jwk
	}
	return jwkMap
}

func getJson(url string, target interface{}) error {
	var myClient = &http.Client{Timeout: 10 * time.Second}
	r, err := myClient.Get(url)
	if err != nil {
		return err
	}
	defer r.Body.Close()

	return json.NewDecoder(r.Body).Decode(target)
}

func validateIdToken(tokenStr, region, userPoolId string,
	jwk map[string]JWKKey) (*jwt.Token, User, error) {
	// Initialize userInfo
	userInfo := User{
		Id:           "",
		Email:        "",
		Group:        "",
		RefreshToken: "",
		Projects:     []string{""},
	}
	// Decode the token string into JWT format.
	token, err := jwt.Parse(tokenStr, func(token *jwt.Token) (interface{},
		error) {
		// cognito user pool : RS256
		if _, ok := token.Method.(*jwt.SigningMethodRSA); !ok {
			return nil, fmt.Errorf("Unexpected signing method: %v",
				token.Header["alg"])
		}
		// Get the kid from the JWT token header and retrieve
		// the corresponding JSON Web Key that was stored
		if kid, ok := token.Header["kid"]; ok {
			if kidStr, ok := kid.(string); ok {
				key := jwk[kidStr]
				// Verify the signature of the decoded JWT token.
				rsaPublicKey := convertKey(key.E, key.N)
				return rsaPublicKey, nil
			}
		}
		// retrieved rsa public key
		return "", nil
	})

	if err != nil {
		return token, userInfo, err
	}

	// fetch fields from token and save into userInfo
	claims := token.Claims.(jwt.MapClaims)

	// id
	sub, ok := claims["sub"]
	if !ok {
		return token, userInfo, fmt.Errorf("token does not have sub attribute")
	}
	id := fmt.Sprint(sub)

	/* set group as worker by default, only assign group to be admin
	when the user has group attribute and the user's group is admin
	*/
	groups, ok := claims["cognito:groups"]
	group := fmt.Sprint(groups)
	if ok && group == "[admin]" { // if admin
		group = group[1 : len(group)-1]
	} else { // by default, users are worker
		group = "worker"
	}

	// email
	Email, ok := claims["email"]
	if !ok {
		return token, userInfo,
			fmt.Errorf("token does not have email attribute")
	}
	emailaddress := fmt.Sprint(Email)

	userInfo = User{
		Id:           id,
		Email:        emailaddress,
		Group:        group,
		RefreshToken: "",
		Projects:     []string{}, // initialize to be empty
	}

	iss, ok := claims["iss"]
	if !ok {
		return token, userInfo, fmt.Errorf("token does not contain issuer")
	}
	issStr := iss.(string)
	if strings.Contains(issStr, "cognito-idp") {
		// 3. 4. 7. Together with checking
		err = validateAWSJwtClaims(claims, region, userPoolId)
		if err != nil {
			return token, userInfo, err
		}
	}

	if token.Valid {
		return token, userInfo, nil
	}
	return token, userInfo, err
}

// Verify the refreshToken feteched from cookie by id as the key
func verifyRefreshToken(refreshToken string, id string) bool {
	// check if any of them is empty, return false
	if (id == "") || (refreshToken == "") {
		return false
	}
	/* TODO: check if refresh token is expired,
	   remove refresh token from backend (and cookie?) */

	// fetech correctRefreshToken saved in our memory
	correctRefreshToken := Users[id].RefreshToken
	if correctRefreshToken != "" {
		return correctRefreshToken == refreshToken
	}
	return false
}

// Print the error message passed in and redirect to login page
func redirectToLogin(w http.ResponseWriter, r *http.Request,
	errorMessage string) {
	Error.Println(errors.New(errorMessage))
	Info.Println("redirect to login")
	authUrl := fmt.Sprintf("https://%v.auth.%v.amazoncognito.com/login?",
		env.DomainName, env.Region) +
		fmt.Sprintf("response_type=code&client_id=%v&redirect_uri=%v",
			env.ClientId, env.RedirectUri)
	http.Redirect(w, r, authUrl, 301)
}

func requestToken(w http.ResponseWriter, r *http.Request, clientId string,
	redirectUri string, awsTokenUrl string, code string,
	secret string) (string, string, string) {
	// create request form
	var tokenRequest http.Request
	err := tokenRequest.ParseForm()
	if err != nil {
		Error.Println(err)
	}
	tokenRequest.Form.Add("grant_type", "authorization_code")
	tokenRequest.Form.Add("client_id", clientId)
	tokenRequest.Form.Add("code", code)
	tokenRequest.Form.Add("redirect_uri", redirectUri)
	tokenRequestBody := strings.NewReader(tokenRequest.Form.Encode())

	var req *http.Request
	// Post token request to AWS Cognito to get JWT
	req, err = http.NewRequest("POST", awsTokenUrl, tokenRequestBody)
	if err != nil {
		log.Println(err)
		redirectToLogin(w, r, "Failed to post token request")
		return "", "", ""
	}
	req.Header.Add("Content-Type", "application/x-www-form-urlencoded")

	autString := clientId + ":" + secret
	authorization := b64.URLEncoding.EncodeToString([]byte(autString))
	req.Header.Add("Authorization", "Basic "+authorization)
	clt := http.Client{}
	resp, err := clt.Do(req)
	if err != nil {
		log.Println(err)
		redirectToLogin(w, r, "Failed to post token request")
		return "", "", ""
	}

	// retrieve tokens from responce
	content, err := ioutil.ReadAll(resp.Body)
	// close response body to avoid resource leak
	defer resp.Body.Close()
	if err != nil {
		redirectToLogin(w, r, "Failed to get tokens")
		return "", "", ""
	}
	respBody := string(content)
	s := strings.Split(respBody, ",")
	// check if we get valid tokens:
	if len(s) < 3 {
		redirectToLogin(w, r, "Invalid token format")
		return "", "", ""
	}
	idtoken := s[0]      // s[0] is id token
	accesstoken := s[1]  // s[1] is accesstoken
	refreshtoken := s[2] // s[2] is refreshtoken

	idTokenString := string(strings.Split(idtoken, ":")[1])
	idTokenString = idTokenString[1 : len(idTokenString)-1]

	accessTokenString := string(strings.Split(accesstoken, ":")[1])
	accessTokenString = accessTokenString[1 : len(accessTokenString)-1]

	refreshTokenString := string(strings.Split(refreshtoken, ":")[1])
	refreshTokenString = refreshTokenString[1 : len(refreshTokenString)-1]

	return idTokenString, accessTokenString, refreshTokenString
}

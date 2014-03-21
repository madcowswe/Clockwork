#ifndef  bitecoin_endpoint_server_hpp
#define  bitecoin_endpoint_server_hpp

#include <cstdio>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>

#include <vector>
#include <memory>

#include "bitecoin_protocol.hpp"

#include "bitecoin_hashing.hpp"

namespace bitecoin{

class EndpointServer
	: public Endpoint
{
private:
	EndpointServer(EndpointServer &); // = delete;
	void operator =(const EndpointServer &); // = delete;

	uint32_t m_protocol;
	std::string m_exchangeId, m_serverId;
	std::string m_clientId, m_minerId;

	void CheckSubmission(const Packet_ServerBeginRound *pBeginRound, const submission_t &subClient)
	{
		Log(Log_Debug, "Starting to re-hash data.\n");
		bigint_t correct=HashReference(pBeginRound,subClient.solution.size(), &subClient.solution[0]);
		Log(Log_Debug, "Rehash done.\n");
		
		if(memcmp(correct.limbs, subClient.proof, BIGINT_LENGTH)){
			Log(Log_Error, "CheckSubmission for submission from %s, proof is not correct.", subClient.clientId.c_str());
			throw std::runtime_error("CheckSubmission - Proof is not correct.");
		}
		
		Log(Log_Debug, "Hash verified.");
	}
public:
	
	EndpointServer(
			std::string exchangeId,
			std::string serverId,
			std::unique_ptr<Connection> &conn,
			int logLevel=1
		)
		: Endpoint(conn, std::make_shared<LogDest>(exchangeId, logLevel))
		, m_exchangeId(exchangeId)
		, m_serverId(serverId)
	{}
		
	void Run()
	{
		try{
			Log(Log_Info, "Waiting for client, exchangeId=%s, serverId=%s\n", m_exchangeId.c_str(), m_serverId.c_str());
			auto beginConnect=RecvPacket<Packet_ClientBeginConnect>();
			m_clientId=beginConnect->clientId;
			m_minerId=beginConnect->minerId;
			
			Log(Log_Info, "Received connection from clientId=%s, minerId=%s\n", m_clientId.c_str(), m_minerId.c_str());		
			
			auto completeConnect = std::make_shared<Packet_ServerCompleteConnect>(m_exchangeId, m_serverId);
			SendPacket(completeConnect);
			
			Log(Log_Verbose, "Connected to client.");
			
			uint64_t roundId=0;//1234;
			
			while(1){
				Log(Log_Info, "Starting round %llu.", roundId);

				auto beginRound=std::make_shared<Packet_ServerBeginRound>();
				beginRound->roundId=roundId;
				beginRound->roundSalt=0;//0x1234beef;//0x20ad1d2f;//0;//rand();
				beginRound->chainData.resize(16+(rand()%1000));
				beginRound->maxIndices=8;
				memset(beginRound->c, 0, BIGINT_LENGTH/2);
				// These are just arbitrary values. The real exchange may choose
				// different ones
				beginRound->c[0] = 0xfffff58d;//0xfffff58d;//0x0baddeed;//0xd58065ff;//4294964621;
				beginRound->c[1] = 0xffffffff;//4294967295;//0x1337feed;//0xabbb1275;//4294967295;
				beginRound->c[2] = 0xcbc2b3ff;// 3418534911;//0x600ddead;//0xc224d8ce;//3418534911;
				beginRound->c[3] = 0x7f7d467a;// 2138916474;//0xc0debeef;//0xb85422a0;//2138916474;
				// Again exchange might choose differently
				beginRound->hashSteps= 23;//16+rand()%16;

				Log(Log_Verbose, "Sending chain data.\n");
				SendPacket(beginRound);
				
				auto requestBid=std::make_shared<Packet_ServerRequestBid>();

				double roundLength=(rand()+1.0)/RAND_MAX;
				//roundLength=-log(roundLength)*2.75+0.25;
				roundLength=-log(roundLength)*5+0.25;
				roundLength=std::max(0.25, std::min(60.0, roundLength));

				//Testing override
				//roundLength = 10;//30;
				
				timestamp_t start=now();
				timestamp_t finish=uint64_t(start+roundLength*1e9);
				
				assert(roundLength>=0.0);
				assert(roundLength<=60.0);
				
				requestBid->timeStampRequestBids=start;
				requestBid->timeStampReceiveBids=finish;
				
				SendPacket(requestBid);
				Log(Log_Verbose, "Requested bids.\n");

				auto bid=RecvPacket<Packet_ClientSendBid>();
				timestamp_t timeRecv=now();
				Log(Log_Verbose, "Received bid.\n");
				
				if(timeRecv > finish){
					Log(Log_Info, "Client bid too late.\n");
				}
				
				submission_t subClient;
				subClient.clientId=m_clientId.c_str();
				subClient.solution = bid->solution;
				memcpy(subClient.proof, bid->proof, BIGINT_LENGTH);
				subClient.timeSent=bid->timeSent;
				subClient.timeRecv=timeRecv;
				
				CheckSubmission(beginRound.get(), subClient);
				
				auto summary=std::make_shared<Packet_ServerCompleteRound>();
				summary->roundId=roundId;
				summary->winner=subClient;
				summary->submissions.push_back(subClient);
				
				SendPacket(summary);
				Log(Log_Info, "Round complete.\n");
					
				roundId++;
			}
		}catch(std::exception &e){
			Log(Log_Fatal, "Exception : %s.\n", e.what());
			throw;
		}
	}
};

}; // bitecoin

#endif

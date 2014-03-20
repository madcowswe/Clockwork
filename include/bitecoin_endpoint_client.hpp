#ifndef  bitecoin_endpoint_client_hpp
#define  bitecoin_endpoint_client_hpp

#include <cstdio>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>

#include <vector>
#include <memory>
#include <map>

#include "bitecoin_protocol.hpp"
#include "bitecoin_endpoint.hpp"
#include "bitecoin_hashing.hpp"
//#include "Clockwork.hpp"

#include <random>

//#define USECUDA

#ifdef USECUDA
#include <cuda_runtime.h>
int testcuda();

//template <unsigned N>
void Clockwork_wrapper(uint32_t* staticbank,
						  uint32_t* regbank,
						  uint32_t* sharedbank2,
						  uint32_t* sharedbank1,
						  //uint32_t N,
						  int* bestiBuff,
						  int* bestiBuffHead,
						  int blocks,
						  int threadsPerBlock);

#endif

namespace bitecoin{

	struct metapoint
	{
		uint32_t lower_index;
		uint64_t value;
	};

class EndpointClient
	: public Endpoint
{
private:
	EndpointClient(EndpointClient &);// = delete; //breaks VS
	void operator =(const EndpointClient &);// = delete; //breaks VS

	std::string m_minerId, m_clientId;

	unsigned m_knownRounds;
	std::map<std::string,unsigned> m_knownCoins;
public:
	
	EndpointClient(
			std::string clientId,
			std::string minerId,
			std::unique_ptr<Connection> &conn,
			std::shared_ptr<ILog> &log
		)
		: Endpoint(conn, log)
		, m_minerId(minerId)
		, m_clientId(clientId)
		, m_knownRounds(0)
	{}
		
	/* Here is a default implementation of make bid.
		I would suggest that you override this method as a starting point.
	*/
	virtual void MakeBid(
		const std::shared_ptr<Packet_ServerBeginRound> roundInfo,	// Information about this particular round
		const std::shared_ptr<Packet_ServerRequestBid> request,		// The specific request we received
		double period,																			// How long this bidding period will last
		double skewEstimate,																// An estimate of the time difference between us and the server (positive -> we are ahead)
		std::vector<uint32_t> &solution,												// Our vector of indices describing the solution
		uint32_t *pProof																		// Will contain the "proof", which is just the value
	){
		double tSafetyMargin=0.5;	// accounts for uncertainty in network conditions
		tSafetyMargin+=0.5;	// latency of banked algo
		/* This is when the server has said all bids must be produced by, plus the
			adjustment for clock skew, and the safety margin
		*/
		double tFinish=request->timeStampReceiveBids*1e-9 + skewEstimate - tSafetyMargin;
		
		Log(Log_Verbose, "MakeBid - start, total period=%lg.", period);

		bigint_t point_preload = PoolHashPreload(roundInfo.get());
		//bigint_t point_preload = PoolHashPreload_Nonbroken(roundInfo.get());


		//TODO: weak Seen set & strong GoldenDiff cache

		unsigned Ngd = 1<<(16+1);
		std::vector<std::pair<uint64_t, uint32_t>> pointidxbank(Ngd);

		std::random_device seeder;
		std::minstd_rand rand_engine(seeder());
		std::uniform_int_distribution<uint32_t> uniform_distr;

		auto fastrand = [&]{
			return uniform_distr(rand_engine);
		};

		double difftic = now()*1e-9;

		for (unsigned i = 0; i < Ngd; i++)
		{
			uint32_t curridx = fastrand();
			bigint_t point = point_preload;
			point.limbs[0] = curridx;

			for(unsigned j=0;j<roundInfo->hashSteps;j++){
				PoolHashStep(point, roundInfo.get());
			}

			uint64_t point64 = ((uint64_t)point.limbs[7] << 32) + point.limbs[6];
			pointidxbank[i] = std::make_pair(point64, curridx);

		}

		double diffgent = now()*1e-9;

		std::sort(pointidxbank.begin(), pointidxbank.end());

		uint32_t GoldenDiff = 0;
		uint64_t bestdistance = -1;
		unsigned skipcount = 0;
		unsigned overloadcount = 0;
		for (unsigned i = 0; i < Ngd-1u; i++)
		{
			uint32_t aidx = pointidxbank[i].second;
			uint32_t bidx = pointidxbank[i+1].second;
			if (aidx == bidx)
			{
				skipcount++;
				continue;
			}

			uint64_t a = pointidxbank[i].first;
			uint64_t b = pointidxbank[i+1].first;
			uint64_t currabsdiff;
			if (a>b)
				currabsdiff = a-b;
			else
				currabsdiff = b-a;

			if (currabsdiff <= bestdistance)
			{
				if (currabsdiff == bestdistance){
					overloadcount++;
				} else {
					overloadcount = 0;
					bestdistance = currabsdiff;
					if (aidx > bidx)
						GoldenDiff = aidx - bidx;
					else
						GoldenDiff = bidx - aidx;
				}
			}
		}

		double diffsortscant = now()*1e-9;
		Log(Log_Verbose, "Diff generate: %g\t sort-scan: %g", diffgent-difftic, diffsortscant-diffgent);

		//quick and dirty 2 idx solution in case we run out of time
		uint32_t bsInitVSsucks[] = {0, GoldenDiff};
		std::vector<uint32_t> bestSolution(bsInitVSsucks, bsInitVSsucks+2); //= {0, GoldenDiff};
		bigint_t pointa = pointFromIdx(roundInfo.get(), point_preload, 0);
		bigint_t pointb = pointFromIdx(roundInfo.get(), point_preload, GoldenDiff);
		bigint_t bestProof;
		wide_xor(8, bestProof.limbs, pointa.limbs, pointb.limbs);
		unsigned k = 2;

		Log(Log_Verbose, "Best distance 0x%016x\t GoldenDiff 0x%08x. Skipped %u identical, Overload %u.", bestdistance, GoldenDiff, skipcount, overloadcount);
		
		unsigned nTrials=0;
		double t=now()*1e-9;	// Work out where we are against the deadline
		double timeBudget=tFinish-t;
		static double hashrate = 1<<16;
		double timeBudgetInital = timeBudget;

		while(1){
			unsigned Nss = 0.8 * std::max(timeBudget,0.) * hashrate/roundInfo->hashSteps;
			if (Nss == 0)
			{
				break;
			}
			++nTrials;

			double tic = now()*1e-9;
			
			Log(Log_Debug, "Trial %d.", nTrials);

			unsigned diff = GoldenDiff;//0x94632009;
			std::vector<std::pair<std::pair<uint64_t, uint64_t>, uint32_t>> metapointidxbank;
			metapointidxbank.reserve(Nss);


			std::uniform_int_distribution<uint32_t> uniform_baserange(0u, (uint32_t)(-1) - diff);
			unsigned failcount = 0;
			while (metapointidxbank.size() < Nss)
			{
				uint32_t idx1 = uniform_baserange(rand_engine);
				bigint_t point1 = pointFromIdx(roundInfo.get(), point_preload, idx1);

				uint32_t idx2 = idx1 + diff;
				bigint_t point2 = pointFromIdx(roundInfo.get(), point_preload, idx2);

				bigint_t metapoint;
				wide_xor(8, metapoint.limbs, point1.limbs, point2.limbs);

				if (metapoint.limbs[7] == 0u || failcount >= 0.3*Nss)
				{
					metapointidxbank.push_back(std::make_pair(
						std::make_pair(
						((uint64_t)metapoint.limbs[6] << 32) + metapoint.limbs[5],
						((uint64_t)metapoint.limbs[4] << 32) + metapoint.limbs[3]),
						idx1) );
				} else {
					failcount++;
				}
			}

			if (failcount > 0.20*Nss){
				Log(Log_Verbose, "We failed to clear MSW %d times when filling Nss=%d", failcount, Nss);
				if (failcount >= 0.3*Nss){
					Log(Log_Verbose, "Second pass: Not enough MSW clear: Override!!!!");
				}
			}

			std::sort(metapointidxbank.begin(), metapointidxbank.end());

			std::pair<uint64_t, uint64_t> bestmmpoint = std::make_pair(-1ull, -1ull);
			std::pair<uint32_t, uint32_t> besti;
			bool bestivalid = 0;
			unsigned skipcount = 0;
			unsigned overloadcount = 0;
			for (unsigned i = 0; i < Nss-1u; i++)
			{
				uint32_t aidx = metapointidxbank[i].second;
				uint32_t bidx = metapointidxbank[i+1].second;
				if (aidx == bidx || aidx == bidx + diff || aidx + diff == bidx)
				{
					skipcount++;
					continue;
				}

				std::pair<uint64_t, uint64_t> a = metapointidxbank[i].first;
				std::pair<uint64_t, uint64_t> b = metapointidxbank[i+1].first;
				std::pair<uint64_t, uint64_t> currmmpoint = pairwise_xor(a,b);

				if (currmmpoint <= bestmmpoint)
				{
					if (currmmpoint == bestmmpoint)
					{
						overloadcount++;
					} else {
						bestmmpoint = currmmpoint;
						besti = std::make_pair(aidx, bidx);
						bestivalid = 1;
					}
				}
			}

			Log(Log_Debug, "Second pass: Skipped %u inclusive idx, Overload %u", skipcount, overloadcount);

			if (!bestivalid)
			{
				break;
			}

			uint32_t bestidx[4] = { besti.first, besti.first + diff, besti.second, besti.second + diff};
			std::sort(bestidx, bestidx+4);
			
			bigint_t proof = HashReferencewPreload(roundInfo.get(), point_preload, 4, bestidx);

			//Number of idx
			k = 4;

			double score=wide_as_double(BIGINT_WORDS, proof.limbs);
			Log(Log_Debug, "    Score=%lg", score);
			
			if(wide_compare(BIGINT_WORDS, proof.limbs, bestProof.limbs)<0){
				double leadingzeros = 256 - log(score) * 1.44269504088896340736; //log2(e)
				Log(Log_Verbose, "    Found new best, nTrials=%d, score=%lg, leading zeros=%lg.", nTrials, score, leadingzeros);
				std::vector<uint32_t> resvec(bestidx, bestidx+k);
				bestSolution = resvec;
				bestProof=proof;
			}
			
			double toc=now()*1e-9;	// Work out where we are against the deadline
			if (timeBudget >= 0.4*timeBudgetInital)
			{
				hashrate = (Nss*roundInfo->hashSteps)/(std::max(toc-tic, 0.1));
				Log(Log_Verbose, "New hashrate %g.", hashrate);
			}

			timeBudget=tFinish-toc;
			Log(Log_Debug, "Finish trial %d, time remaining =%lg seconds.", nTrials, timeBudget);
			
			if(timeBudget<=0)
				break;	// We have run out of time, send what we have
		}

		Log(Log_Verbose, "Did %d trials in the end.", nTrials);

		//Trollface
		//wide_zero(8, bestProof.limbs);
		//bestSolution = std::vector<uint32_t>();
		
		solution=bestSolution;
		wide_copy(BIGINT_WORDS, pProof, bestProof.limbs);
		
		Log(Log_Verbose, "MakeBid - finish.");
	}
		
	void Run()
	{
		try{
			#ifdef USECUDA
			Log(Log_Info, "Testing CUDA");
			testcuda();
			#endif

			auto beginConnect=std::make_shared<Packet_ClientBeginConnect>(m_clientId, m_minerId);
			Log(Log_Info, "Connecting with clientId=%s, minerId=%s", m_clientId.begin(), m_minerId.begin());
			SendPacket(beginConnect);
			
			auto endConnect=RecvPacket<Packet_ServerCompleteConnect>();
			Log(Log_Info, "Connected to exchange=%s, running=%s", endConnect->exchangeId.c_str(), endConnect->serverId.c_str());
			
			while(1){
				Log(Log_Verbose, "Waiting for round to begin.");
				auto beginRound=RecvPacket<Packet_ServerBeginRound>();
				Log(Log_Info, "Round beginning with %u bytes of chain data.", beginRound->chainData.size());
				Log(Log_Info, "ID: %u\tMaxIdx: %u\tSteps: %u\tSalt: %#x\tc: 0x%08x%08x%08x%08x", beginRound->roundId, beginRound->maxIndices, beginRound->hashSteps, beginRound->roundSalt, beginRound->c[3], beginRound->c[2], beginRound->c[1], beginRound->c[0]);
				

				Log(Log_Verbose, "Waiting for request for bid.");
				auto requestBid=RecvPacket<Packet_ServerRequestBid>();
				// Get an estimate of the skew between our clock and theirs. If it is positive,
				// then we are ahead of them.
				double tNow=now()*1e-9;
				double skewEstimate=tNow - requestBid->timeStampRequestBids*1e-9;
				// And work out how long they expect it to last, independent of the skew
				double period=requestBid->timeStampReceiveBids*1e-9 - requestBid->timeStampRequestBids*1e-9;
	
				Log(Log_Info, "Received bid request: serverStart=%lf, ourStart=%lf, skew=%lg. Bid period=%lf", requestBid->timeStampRequestBids*1e-9,  tNow, skewEstimate, period);
				
				std::shared_ptr<Packet_ClientSendBid> bid=std::make_shared<Packet_ClientSendBid>();
				
				MakeBid(beginRound, requestBid, period, skewEstimate, bid->solution, bid->proof);
				if(bid.use_count() == 0)
					assert(false);
				bid->timeSent=now();
				Log(Log_Verbose, "Bid ready.");

				for (unsigned i = 0; i < bid->solution.size(); ++i)
				{
					Log(Log_Debug, "    Indicies were 0x%08x", bid->solution[bid->solution.size()-1-i]);
				}

				if (bid->solution.size() >= 2)
				{
					Log(Log_Debug, "    Diff is       0x%08x", bid->solution[1] - bid->solution[0]);
				}
				Log(Log_Debug, "    MSW is        0x%08x", bid->proof[7]);

				SendPacket(bid);
				Log(Log_Verbose, "Bid sent.");
				
				Log(Log_Verbose, "Waiting for results.");
				auto results=RecvPacket<Packet_ServerCompleteRound>();
				Log(Log_Info, "Got round results.");

				static unsigned overduectr = 0;
				
				for(unsigned i=0;i<results->submissions.size();i++){
					double taken=requestBid->timeStampReceiveBids*1e-9 - results->submissions[i].timeRecv*1e-9;
					bool overDue=requestBid->timeStampReceiveBids < results->submissions[i].timeRecv;
					if (overDue && results->submissions[i].clientId.compare(m_clientId) == 0)
						overduectr++;
					Log(Log_Info, "  %16s : %.6lg, %lg%s", results->submissions[i].clientId.c_str(),
							wide_as_double(BIGINT_WORDS, results->submissions[i].proof), taken,
							overDue?" OVERDUE":""
					);
					if(m_knownCoins.find(results->submissions[i].clientId)==m_knownCoins.end()){
						m_knownCoins[results->submissions[i].clientId]=0;
					}
				}

				Log(Log_Info, "We have been overdue %d times.", overduectr);
				
				if(results->winner.clientId==m_clientId){
					Log(Log_Info, "");
					Log(Log_Info, "You won a coin!");
					Log(Log_Info, "");
				}
				
				m_knownRounds++;
				m_knownCoins[results->winner.clientId]++;
				
				Log(Log_Info, "  %16s : %6s, %8s\n", "ClientId", "Coins", "Success");
				auto it=m_knownCoins.begin();
				while(it!=m_knownCoins.end()){
					Log(Log_Info, "  %16s : %6d, %.6lf", it->first.c_str(), it->second, it->second/(double)m_knownRounds);
					++it;
				}

				
				Log(Log_Verbose, "");
			}

		}catch(std::exception &e){
			Log(Log_Fatal, "Exception : %s.", e.what());
			throw;
		}
	}
};

}; // bitecoin

#endif


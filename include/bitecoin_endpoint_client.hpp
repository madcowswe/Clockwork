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
#include <array>
#include <algorithm>

#include "bitecoin_protocol.hpp"
#include "bitecoin_endpoint.hpp"
#include "bitecoin_hashing.hpp"
#include "wide_int.h"


//#include "Clockwork.hpp"

#include <random>

#include "tbb/parallel_sort.h"
#include "tbb/parallel_for.h"
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
		static std::map<std::pair<uint32_t, std::array<uint32_t, BIGINT_WORDS / 2>>, uint32_t> diffCache;
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

		unsigned Ngd = 1<<(16+2);
		std::vector<std::pair<uint64_t, uint32_t>> pointidxbank(Ngd);

		std::random_device seeder;
		std::minstd_rand rand_engine(seeder());
		std::uniform_int_distribution<uint32_t> uniform_distr;

		

		

		auto fastrand = [&]{
			return uniform_distr(rand_engine);
		};

		double difftic = now()*1e-9;
		std::array<uint32_t, 4> temp = { { roundInfo.get()->c[0], roundInfo.get()->c[1], roundInfo.get()->c[2], roundInfo.get()->c[3] } };
		std::pair<uint32_t, std::array<uint32_t, BIGINT_WORDS / 2>> key = std::make_pair(roundInfo.get()->hashSteps, temp);
		uint32_t GoldenDiff = 0;
		if (diffCache.find(key) == diffCache.end())
		{
			//for (unsigned i = 0; i < Ngd; i++)
			tbb::parallel_for((unsigned)0, Ngd, [&](unsigned i) {
				uint32_t curridx = fastrand();
				bigint_t point = point_preload;
				point.limbs[0] = curridx;

				for (unsigned j = 0; j < roundInfo->hashSteps; j++){
					PoolHashStep(point, roundInfo.get());
				}

				uint64_t point64 = ((uint64_t)point.limbs[7] << 32) + point.limbs[6];
				pointidxbank[i] = std::make_pair(point64, curridx);

			});

			double diffgent = now()*1e-9;

			tbb::parallel_sort(pointidxbank.begin(), pointidxbank.end());
						
			uint64_t bestdistance = -1;
			//unsigned skipcount = 0;
			unsigned overloadcount = 0;
			for (unsigned i = 0; i < Ngd - 1u; i++)
			{
				uint32_t aidx = pointidxbank[i].second;
				uint32_t bidx = pointidxbank[i + 1].second;
				if (aidx == bidx)
				{
					//skipcount++;
					continue;
				}

				uint64_t a = pointidxbank[i].first;
				uint64_t b = pointidxbank[i + 1].first;
				uint64_t currabsdiff;
				if (a > b)
					currabsdiff = a - b;
				else
					currabsdiff = b - a;

				if (currabsdiff <= bestdistance)
				{
					if (currabsdiff == bestdistance){
						overloadcount++;
					}
					else {
						overloadcount = 0;
						bestdistance = currabsdiff;
						if (aidx > bidx)
							GoldenDiff = aidx - bidx;
						else
							GoldenDiff = bidx - aidx;
					}
				}
			}


			diffCache[key] = GoldenDiff;
			double diffsortscant = now()*1e-9;
			Log(Log_Verbose, "Diff generate: %g\t sort-scan: %g", diffgent - difftic, diffsortscant - diffgent);

		}
		else 
		{
			GoldenDiff = diffCache[key];
			Log(Log_Verbose, "\n\n\nFOUND DIFF! %d\n\n\n", GoldenDiff);
		}




		//quick and dirty 2 idx solution in case we run out of time
		uint32_t bsInitVSsucks[] = {0, GoldenDiff};
		std::vector<uint32_t> bestSolution(bsInitVSsucks, bsInitVSsucks+2); //= {0, GoldenDiff};
		bigint_t pointa = pointFromIdx(roundInfo.get(), point_preload, 0);
		bigint_t pointb = pointFromIdx(roundInfo.get(), point_preload, GoldenDiff);
		bigint_t bestProof;
		wide_xor(8, bestProof.limbs, pointa.limbs, pointb.limbs);
		unsigned k = 2;

	//	Log(Log_Verbose, "Best distance 0x%016x\t GoldenDiff 0x%08x. Skipped %u identical, Overload %u.", bestdistance, GoldenDiff, skipcount, overloadcount);
		
		unsigned nTrials=0;
		double t=now()*1e-9;	// Work out where we are against the deadline
		double timeBudget=tFinish-t;
		static double hashrate = 1<<16;
		static double avgHR = 0;
		static int nSample = 0;
		double timeBudgetInital = timeBudget;

		unsigned maxIdx = roundInfo.get()->maxIndices;

		while(1){
			unsigned Nss = 0.8 * std::max(timeBudget,0.) * hashrate/roundInfo->hashSteps;
			if (Nss == 0)
			{
				break;
			}
			++nTrials;

			double tic = now()*1e-9;
			
			Log(Log_Debug, "Trial %d.", nTrials);

			int enabledIndicies = 2;
			std::array<uint32_t, 16> besti;


			if (maxIdx >= 4) 
			{
				enabledIndicies = 4;
				std::vector<wide_idx_pair_4> M1pointIdxBank;
				std::vector<wide_idx_pair_4> M2pointIdxBank;
				std::vector<wide_idx_pair_4> M3pointIdxBank;
				std::vector<wide_idx_pair_4>* currentBank = &M1pointIdxBank;


				//2 depth:	Generate indicies
				//			Generate points from indicies
				//			XOR to make meta-points
				//			Put metapoints and indicies into bank (1 base 1 implied)
				//			Sort
				//3 depth:	Take meta-points and indicies
				//			XOR to make meta-meta points store indicies (2 base - 2 implied)
				//			Sort
				//4 depth:	Take meta-meta points and indicies 4 base - 4 implied
				//			XOR to make meta^3-points
				//			Sort
				//

				//gen
				unsigned diff = GoldenDiff;//0x94632009;
				std::uniform_int_distribution<uint32_t> uniform_baserange(0u, (uint32_t)(-1) - diff);
				//M1pointIdxBank.reserve(Nss);
				M1pointIdxBank.resize(Nss);
				//for (unsigned i = 0; i < Nss; i++)
				tbb::parallel_for((unsigned)0, Nss, [&](unsigned i)	
				{
					uint32_t idx1 = uniform_baserange(rand_engine);
					bigint_t point1 = pointFromIdx(roundInfo.get(), point_preload, idx1);

					uint32_t idx2 = idx1 + diff;
					bigint_t point2 = pointFromIdx(roundInfo.get(), point_preload, idx2);

					bigint_t metapoint;
					wide_xor(8, metapoint.limbs, point1.limbs, point2.limbs);

					wide_idx_pair_4 newMetapoint;
					
					newMetapoint.first.first = std::make_pair(
						((uint64_t)metapoint.limbs[7] << 32) + metapoint.limbs[6],
						((uint64_t)metapoint.limbs[5] << 32) + metapoint.limbs[4]);

					newMetapoint.first.second = std::make_pair(
						((uint64_t)metapoint.limbs[3] << 32) + metapoint.limbs[2],
						((uint64_t)metapoint.limbs[1] << 32) + metapoint.limbs[0]);

					newMetapoint.second[0] = idx1;

					//M1pointIdxBank.push_back(newMetapoint);
					M1pointIdxBank[i] = newMetapoint;
				});

				double tic2 = now()*1e-9;
				if((tic2 - tic) > 0.1*timeBudgetInital)
					Log(Log_Verbose, "gen :%g", (tic2 - tic));
				else
					Log(Log_Debug, "gen :%g", (tic2 - tic));

				//sort
				tbb::parallel_sort(M1pointIdxBank.begin(), M1pointIdxBank.end());

				int workingBankSize = std::max((int)M1pointIdxBank.size() - 1, 0);


				do { //construct for using break to get to end
					if (maxIdx < 8) 
						break;

					//M2pointIdxBank.reserve(workingBankSize);
					M2pointIdxBank.resize(workingBankSize);
					currentBank = &M2pointIdxBank;
					enabledIndicies = 8;
					//unsigned skipcount = 0;
					
					//Depth 3:
					//for (int i = 0; i < workingBankSize; i++)
					tbb::parallel_for((int)0, workingBankSize, [&](int i)				
					{
						
						uint32_t aidx = M1pointIdxBank[i].second[0];
						uint32_t bidx = M1pointIdxBank[i + 1].second[0];

						std::array<uint32_t, 4> indicies = { aidx, bidx, aidx + diff, bidx + diff };
						std::sort(indicies.begin(), indicies.end());
						auto x = std::adjacent_find(indicies.begin(), indicies.end());

						if (x == indicies.end())
						{
						//	//Log(Log_Verbose, "Skipped index:%d", i);
						//	//skipcount++;
						//	continue;
						//}
						//else {

							auto a = M1pointIdxBank[i].first;
							auto b = M1pointIdxBank[i + 1].first;
							auto currmmpoint = wap_xor(a, b);

							wide_idx_pair_4 wip;

							//Meta-meta points
							wip.first = currmmpoint;

							//Update indicies
							wip.second[0] = aidx;
							wip.second[1] = bidx;

							//M2pointIdxBank.push_back(wip);
							M2pointIdxBank[i] = wip;
						}
					});
					//Log(Log_Debug, "Second loop. Skipped %d", skipcount);
					tbb::parallel_sort(M2pointIdxBank.begin(), M2pointIdxBank.end());
					
					workingBankSize = std::max((int)M2pointIdxBank.size() - 1, 0);



					//Depth 4:
					if (maxIdx < 16) 
						break;

					//unsigned skipcount1 = 0;
					enabledIndicies = 16;
					currentBank = &M3pointIdxBank;
					//M3pointIdxBank.reserve(workingBankSize);
					M3pointIdxBank.resize(workingBankSize);

					//for (int i = 0; i < workingBankSize; i++)
					tbb::parallel_for((int)0, workingBankSize, [&](int i)
					{
						uint32_t aidx1 = M2pointIdxBank[i].second[0];
						uint32_t aidx2 = M2pointIdxBank[i].second[1];
						uint32_t bidx1 = M2pointIdxBank[i + 1].second[0];
						uint32_t bidx2 = M2pointIdxBank[i + 1].second[1];

						std::array<uint32_t, 8> indicies = { aidx1, aidx2, bidx1, bidx2, aidx1 + diff, aidx2 + diff, bidx1 + diff, bidx2 + diff };
						std::sort(indicies.begin(), indicies.end());
						auto x = std::adjacent_find(indicies.begin(), indicies.end());

						if (x == indicies.end())
						{
							auto a = M2pointIdxBank[i].first;
							auto b = M2pointIdxBank[i + 1].first;
							auto currmmpoint = wap_xor(a, b);

							wide_idx_pair_4 wip;

							//Meta-meta-meta points
							wip.first = currmmpoint;

							//Update indicies
							wip.second[0] = aidx1;
							wip.second[1] = aidx2;
							wip.second[2] = bidx1;
							wip.second[3] = bidx2;

							//M3pointIdxBank.push_back(wip);
							M3pointIdxBank[i] = (wip);
						}

					});

					//Log(Log_Debug, "Third loop. Skipped %d", skipcount1);

					tbb::parallel_sort(M3pointIdxBank.begin(), M3pointIdxBank.end());

					Log(Log_Debug, "Third loop");

					workingBankSize = ((int)M3pointIdxBank.size()) - 1;
				} while (0);

				unsigned overloadcount = 0;
				unsigned skipcount2 = 0;
				wide_as_pair bestmmpoint = std::make_pair(std::make_pair(-1ull, -1ull), std::make_pair(-1ull, -1ull));
				
				bool bestivalid = 0;
				for (int i = 0; i < workingBankSize; i++)
				{
					std::array<uint32_t, 16> indicies;
	
					unsigned idxi = 0;
					for (int j = 0; j < enabledIndicies / 4; j++)
						indicies[idxi++] = (*currentBank)[i].second[j];

					for (int j = 0; j < enabledIndicies / 4; j++)
						indicies[idxi++] = (*currentBank)[i + 1].second[j];

					for (int j = 0; j < enabledIndicies / 4; j++)
						indicies[idxi++] = (*currentBank)[i].second[j] + diff;

					for (int j = 0; j < enabledIndicies / 4; j++)
						indicies[idxi++] = (*currentBank)[i + 1].second[j] + diff;

					std::sort(indicies.begin(), indicies.begin() + enabledIndicies);
					auto x = std::adjacent_find(indicies.begin(), indicies.begin() + enabledIndicies);

					if (x != indicies.begin() + enabledIndicies)
					{
						skipcount2++;
						continue;
					}

					auto currmmpoint = wap_xor((*currentBank)[i].first, (*currentBank)[i + 1].first);

					if (currmmpoint <= bestmmpoint)
					{
						if (currmmpoint == bestmmpoint)
						{
							overloadcount++;
						}
						else {
							bestmmpoint = currmmpoint;
							besti = indicies;
							bestivalid = 1;
						}
					}
				}

				//}
				
				//And now we do meta-meta points
				double tocscan = now()*1e-9;
				if((tocscan - tic2) > 0.1*timeBudgetInital)
					Log(Log_Verbose, "sort-scan :%g", (tocscan - tic2));
				else
					Log(Log_Debug, "sort-scan :%g",  (tocscan - tic2));
				Log(Log_Debug, "Final pass: Skipped %u inclusive idx, Overload %u", skipcount2, overloadcount);
				//Log(Log_Verbose, "\nAmazing tom\n");

				if (!bestivalid)
				{
					break;
				}
			}

			std::sort(besti.begin(), besti.begin() + enabledIndicies);

			bigint_t proof = HashReferencewPreload(roundInfo.get(), point_preload, enabledIndicies, &besti[0]);

			//Number of idx
			k = enabledIndicies;

			double score=wide_as_double(BIGINT_WORDS, proof.limbs);
			Log(Log_Debug, "    Score=%lg", score);
			
			if(wide_compare(BIGINT_WORDS, proof.limbs, bestProof.limbs)<0){
				double leadingzeros = 256 - log(score) * 1.44269504088896340736; //log2(e)
				Log(Log_Verbose, "    Found new best, nTrials=%d, score=%lg, leading zeros=%lg.", nTrials, score, leadingzeros);
				std::vector<uint32_t> resvec(&besti[0], &besti[0] + k);
				bestSolution = resvec;
				bestProof=proof;
			}
			
			double toc=now()*1e-9;	// Work out where we are against the deadline
			if (timeBudget >= 0.4*timeBudgetInital)
			{
				hashrate = (Nss*roundInfo->hashSteps)/(std::max(toc-tic, 0.1));
				avgHR += hashrate;
				nSample++;
				Log(Log_Verbose, "New hashrate %g.", hashrate);
				Log(Log_Verbose, "==Avg hashrate %g.", avgHR / nSample);
			}

			timeBudget=tFinish-toc;
			Log(Log_Debug, "Finish trial %d, time remaining =%lg seconds.", nTrials, timeBudget);
			
			if(timeBudget<=0)
				break;	// We have run out of time, send what we have
		}

		Log(Log_Verbose, "Did %d trials in the end.", nTrials);
		
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


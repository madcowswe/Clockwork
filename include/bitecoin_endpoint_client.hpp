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

//#define HR2BANKED
#define DIFFTHINGY
//#define HR2BANKED_DEGEN
//#define HRkBANKED
//#define HRCUDA

#ifdef HRCUDA
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
		
		/*
			We will use this to track the best solution we have created so far.
		*/
		std::vector<uint32_t> bestSolution(roundInfo->maxIndices);
		bigint_t bestProof;
		wide_ones(BIGINT_WORDS, bestProof.limbs);

		bigint_t point_preload = PoolHashPreload(roundInfo.get());
		//bigint_t point_preload = PoolHashPreload_Nonbroken(roundInfo.get());

		unsigned N = 1<<16;
		std::vector<std::pair<uint64_t, uint32_t>> pointidxbank(N);
		auto fastrand = std::minstd_rand();
		for (unsigned i = 0; i < N; i++)
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

		std::sort(pointidxbank.begin(), pointidxbank.end());

		uint32_t GoldenDiff = 0;
		uint64_t bestdistance = -1;
		unsigned dbgcnt = 0;
		for (unsigned i = 0; i < N-1u; i++)
		{
			uint32_t aidx = pointidxbank[i].second;
			uint32_t bidx = pointidxbank[i+1].second;
			if (aidx == bidx)
			{
				dbgcnt++;
				continue;
			}

			uint64_t a = pointidxbank[i].first;
			uint64_t b = pointidxbank[i+1].first;
			uint64_t currabsdiff;
			if (a>b)
				currabsdiff = a-b;
			else
				currabsdiff = b-a;

			if (currabsdiff < bestdistance)
			{
				bestdistance = currabsdiff;
				if (aidx > bidx)
					GoldenDiff = aidx - bidx;
				else
					GoldenDiff = bidx - aidx;
			}
		}

		Log(Log_Verbose, "Best distance 0x%016x\t GoldenDiff %08x. Skipped %u identical.", bestdistance, GoldenDiff, dbgcnt);
		
		unsigned nTrials=0;
		while(1){
			++nTrials;
			
			Log(Log_Debug, "Trial %d.", nTrials);

#ifdef DIFFTHINGY
			unsigned diff = 0x94632009; //FROM OSKAR
			//unsigned N = 10000;

			std::vector<uint32_t> indices(roundInfo->maxIndices);
			std::vector<uint32_t> idxbanks;
			idxbanks.resize(N);

			//Generate point for each index pair
			std::vector<uint64_t> pointbanks[2];
			pointbanks[0].resize(N);
			pointbanks[1].resize(N);
			//std::vector<metapoint> idx_mpoint_bank;
			//idx_mpoint_bank.resize(N);

			for (unsigned i = 0; i < N; ++i)
			{
				idxbanks[i] = rand() / 2;

				for (unsigned currbank = 0; currbank < 2; ++currbank)
				{
					bigint_t point = point_preload;
					point.limbs[0] = idxbanks[i] + currbank*diff;

					// Now step forward by the number specified by the server
					for (unsigned j = 0; j<roundInfo.get()->hashSteps; j++){
						//Needs to become 64 bit (upper 2 MSW)
						PoolHashStep(point, roundInfo.get());
					}

					pointbanks[currbank][i] = ((uint64_t)point.limbs[7] << 32) + point.limbs[6];
				}

				//XOR point pair to make meta-point and put in bank
				pointidxbank[i].first = pointbanks[1][i] ^ pointbanks[0][i];
				pointidxbank[i].second = idxbanks[i];

			}

			
			//XOR meta-points and find minimum
			uint32_t bestIndex[2];
			uint64_t currentMin = -1;
			uint64_t currentValue;

			for (unsigned i = 0; i < N; ++i)
			{
				for (int j = 0; j < (int) (i - 1); ++j)
				{
					if (i == j)
						assert(false);

					//Check indicies are distinct, if so, skip
					if (pointidxbank[i].second == pointidxbank[j].second || pointidxbank[i].second + diff == pointidxbank[j].second || pointidxbank[j].second + diff == pointidxbank[i].second)
						continue;

					currentValue = pointidxbank[i].first ^ pointidxbank[j].first;

					if (currentValue < currentMin || (currentValue == currentMin && currentValue < currentMin))
					{						
						bestIndex[0] = pointidxbank[i].second;
						bestIndex[1] = pointidxbank[j].second;
						currentMin = currentValue;
						currentMin = currentValue;
					}
				}
			}

			uint32_t bestidx[4] = { bestIndex[0], bestIndex[0] + diff, bestIndex[1], bestIndex[1] + diff};

			std::sort(&bestidx[0], &bestidx[0] + 4);

			//Sort bestidx
			bigint_t proof = HashReferencewPreload(roundInfo.get(), point_preload, 4, bestidx);

			//Number of banks
			unsigned k = 4;


#endif


#ifdef HR2BANKED

			unsigned N = 10000;
			std::vector<uint32_t> idxbanks[2];
			idxbanks[0].reserve(N);
			idxbanks[1].reserve(N);

			for (unsigned i = 0; i < N; ++i)
			{
				idxbanks[0][i] = rand()/2;
			}

			for (unsigned i = 0; i < N; ++i)
			{
				idxbanks[1][i] = rand()/2 + (1<<31);
			}

			unsigned besti, bestj;
			HashReference2Banked(roundInfo.get(), point_preload, N, idxbanks, besti, bestj, 7);

			uint32_t bestidx[2] = {idxbanks[0][bestj], idxbanks[1][besti]};
			bigint_t proof=HashReferencewPreload(roundInfo.get(), point_preload, 2, bestidx);//, m_log);

			unsigned k = 2;
#endif
#ifdef HR2SINGLEBANK
			//TODO: Find golden diff using standard
#endif
#ifdef HR2BANKED_DEGEN

			unsigned N = 10000;
			std::vector<uint32_t> idxbanks[2];
			idxbanks[0].reserve(N);
			idxbanks[1].reserve(N);

			for (unsigned i = 0; i < N; ++i)
			{
				idxbanks[0][i] = rand()/2;
				idxbanks[1][i] = idxbanks[0][i] + 0x94632009;
			}

			unsigned besti, bestj;
			HashReference2Banked(roundInfo.get(), point_preload, N, idxbanks, besti, bestj, 6);

			uint32_t bestidx[2] = {idxbanks[0][bestj], idxbanks[1][besti]};
			bigint_t proof=HashReferencewPreload(roundInfo.get(), point_preload, 2, bestidx);//, m_log);

			unsigned k = 2;
#endif
#ifdef HRkBANKED

			#define kpow 2
			#define ALLOC_k (1u<<kpow)

			unsigned k = std::min((1u<<kpow), roundInfo->maxIndices);
			unsigned N = 1<<(28/k);
			unsigned subspace_size = 1<<(32-kpow);

			std::vector<uint32_t> idxbanks[ALLOC_k];
			for (unsigned i = 0; i < k; ++i)
			{
				idxbanks[i].reserve(N);
			}

			for (unsigned i = 0; i < k; ++i)
			{
				for (unsigned j = 0; j < N; ++j)
				{
					idxbanks[i].push_back((rand() & (subspace_size-1)) + i*subspace_size);
					//fprintf(stderr, "gen: bank %u\ti %u\tval %#x\n", i, j, idxbanks[i][j]);
				}
			}

			unsigned besti[ALLOC_k];
			HashReferencekBanked<ALLOC_k>(roundInfo.get(), point_preload, N, k, idxbanks, besti);

			uint32_t bestidx[ALLOC_k];
			for (unsigned i = 0; i < k; ++i)
			{
				bestidx[i] = idxbanks[i][besti[i]];
				//fprintf(stderr, "select: bank %u\ti %u\tval %#x\n", i, besti[i], bestidx[i]);
			}
			bigint_t proof=HashReferencewPreload(roundInfo.get(), point_preload, k, bestidx);

#endif
#ifdef HRCUDA
			cudaError e;

			#define blocks 6
			#define threadsPerBlock 1024
			#define N 128
			size_t banksizeBytes = N*sizeof(uint32_t);

			uint32_t staticidx[blocks*threadsPerBlock];
			uint32_t regidx[N];
			uint32_t sharedidx2[N];
			uint32_t sharedidx1[N];
			uint32_t staticpoints[blocks*threadsPerBlock];
			uint32_t regpoints[N];
			uint32_t sharedpoints2[N];
			uint32_t sharedpoints1[N];

			unsigned subspace_size = 1<<30;
			for (int i = 0; i < blocks*threadsPerBlock; ++i){
				staticidx[i] = (rand() & (subspace_size-1)) + 3*subspace_size;
			}
			for (int i = 0; i < N; ++i){
				regidx[i] = (rand() & (subspace_size-1)) + 2*subspace_size;
			}
			for (int i = 0; i < N; ++i){
				sharedidx2[i] = (rand() & (subspace_size-1)) + 1*subspace_size;
			}
			for (int i = 0; i < N; ++i){
				sharedidx1[i] = (rand() & (subspace_size-1)) + 0*subspace_size;
			}

			pointsFromIdx(roundInfo.get(), point_preload, blocks*threadsPerBlock, staticidx, staticpoints);
			pointsFromIdx(roundInfo.get(), point_preload, N, regidx, regpoints);
			pointsFromIdx(roundInfo.get(), point_preload, N, sharedidx2, sharedpoints2);
			pointsFromIdx(roundInfo.get(), point_preload, N, sharedidx1, sharedpoints1);

			uint32_t* staticbank_GPU, *regbank_GPU, *sharedbank1_GPU, *sharedbank2_GPU;
			if(e = cudaMalloc(&staticbank_GPU, blocks*threadsPerBlock*sizeof(uint32_t))) fprintf(stderr, "Cuda error %d on line %d\n", e, __LINE__);
			if(e = cudaMalloc(&regbank_GPU, banksizeBytes)) fprintf(stderr, "Cuda error %d on line %d\n", e, __LINE__);
			if(e = cudaMalloc(&sharedbank2_GPU, banksizeBytes)) fprintf(stderr, "Cuda error %d on line %d\n", e, __LINE__);
			if(e = cudaMalloc(&sharedbank1_GPU, banksizeBytes)) fprintf(stderr, "Cuda error %d on line %d\n", e, __LINE__);

			if(e = cudaMemcpy(staticbank_GPU, staticpoints, blocks*threadsPerBlock*sizeof(uint32_t), cudaMemcpyHostToDevice)) fprintf(stderr, "Cuda error %d on line %d\n", e, __LINE__);
			if(e = cudaMemcpy(regbank_GPU, regpoints, banksizeBytes, cudaMemcpyHostToDevice)) fprintf(stderr, "Cuda error %d on line %d\n", e, __LINE__);
			if(e = cudaMemcpy(sharedbank2_GPU, sharedpoints2, banksizeBytes, cudaMemcpyHostToDevice)) fprintf(stderr, "Cuda error %d on line %d\n", e, __LINE__);
			if(e = cudaMemcpy(sharedbank1_GPU, sharedpoints1, banksizeBytes, cudaMemcpyHostToDevice)) fprintf(stderr, "Cuda error %d on line %d\n", e, __LINE__);

			int* bestiBuff_GPU;
			if(e = cudaMalloc(&bestiBuff_GPU, 4*1024*sizeof(int))) fprintf(stderr, "Cuda error %d on line %d\n", e, __LINE__);

			__device__ int theHeadPtr_GPU;
			if(e = cudaMemset(&theHeadPtr_GPU, 0, sizeof(int))) fprintf(stderr, "Cuda error %d on line %d\n", e, __LINE__);
			//int z = 0;
			//cudaMemcpyToSymbol(&theHeadPtr_GPU, &z, sizeof(int));

			cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

			//TODO Make N variable

			Clockwork_wrapper (staticbank_GPU, regbank_GPU, sharedbank2_GPU, sharedbank1_GPU, bestiBuff_GPU, &theHeadPtr_GPU, blocks, threadsPerBlock);


			
			bigint_t proof; // = TODO
			uint32_t bestidx[4]; //TODO, change this!
			unsigned k = 4;
#endif

			double score=wide_as_double(BIGINT_WORDS, proof.limbs);
			Log(Log_Debug, "    Score=%lg", score);
			
			if(wide_compare(BIGINT_WORDS, proof.limbs, bestProof.limbs)<0){
				double worst=pow(2.0, BIGINT_LENGTH*8);	// This is the worst possible score
				Log(Log_Verbose, "    Found new best, nTrials=%d, score=%lg, ratio=%lg.", nTrials, score, worst/score);
				std::vector<uint32_t> resvec(bestidx, bestidx+k);
				bestSolution = resvec;
				bestProof=proof;
			}
			
			double t=now()*1e-9;	// Work out where we are against the deadline
			double timeBudget=tFinish-t;
			Log(Log_Debug, "Finish trial %d, time remaining =%lg seconds.", nTrials, timeBudget);
			
			if(timeBudget<=0)
				break;	// We have run out of time, send what we have
		}
		
		solution=bestSolution;
		wide_copy(BIGINT_WORDS, pProof, bestProof.limbs);
		
		Log(Log_Verbose, "MakeBid - finish.");
	}
		
	void Run()
	{
		try{
			#ifdef HRCUDA
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
				bid->timeSent=now();				
				Log(Log_Verbose, "Bid ready.");

				for (unsigned i = 0; i < bid->solution.size(); ++i)
				{
					Log(Log_Verbose, "    Indicies were 0x%08x", bid->solution[bid->solution.size()-1-i]);
				}

				Log(Log_Verbose, "    Diff is       0x%08x", bid->solution[1] - bid->solution[0]);
				Log(Log_Verbose, "    MSW is        0x%08x", bid->proof[7]);

				SendPacket(bid);
				Log(Log_Verbose, "Bid sent.");
				
				Log(Log_Verbose, "Waiting for results.");
				auto results=RecvPacket<Packet_ServerCompleteRound>();
				Log(Log_Info, "Got round results.");
				
				for(unsigned i=0;i<results->submissions.size();i++){
					double taken=requestBid->timeStampReceiveBids*1e-9 - results->submissions[i].timeRecv*1e-9;
					bool overDue=requestBid->timeStampReceiveBids < results->submissions[i].timeRecv;
					Log(Log_Info, "  %16s : %.6lg, %lg%s", results->submissions[i].clientId.c_str(),
							wide_as_double(BIGINT_WORDS, results->submissions[i].proof), taken,
							overDue?" OVERDUE":""
					);
					if(m_knownCoins.find(results->submissions[i].clientId)==m_knownCoins.end()){
						m_knownCoins[results->submissions[i].clientId]=0;
					}
				}
				
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


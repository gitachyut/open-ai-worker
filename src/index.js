import { ChatOpenAI } from '@langchain/openai';
import { StringOutputParser, JsonOutputParser } from '@langchain/core/output_parsers';
import { PromptTemplate } from '@langchain/core/prompts';

const llm = new ChatOpenAI({
	model: "gpt-4o-mini",
	openAIApiKey: ""
});

const founderSummaryPrompt = PromptTemplate.fromTemplate(`
   You are a experienced VC/Investor Analayser with yeras of experience in analyzing companies for investment opprtunities.Given the company info below , generate a comprehensive two-paragraph summary. Focus on capturing the essence of the business vision, the problem being solved, the founding team's strengths, and their execution strategy. Stick to the facts and do not give opinions.

    Company Info:

   {company_data}
     Try to follow the format below:
      Follow these steps to create the summary:

    1. First Paragraph - Company Overview and Market Position:
       - Begin with company name, legal name, and founding team
       - State the core problem they're addressing and their industry
       - Include their location details and product launch information
       - Describe their competitive positioning and unique value proposition
       - Explain their go-to-market strategy and target market

    2. Second Paragraph - Financial Position and Future Plans:
       - Detail their financial performance (revenue, EBITDA)
       - Include current cash position and burn rate
       - Highlight founding team's achievements and funding experience
       - State their fundraising goals and valuation
       - End with their execution strategy and growth plans

       Key points to remember:
    - Focus on facts, avoid opinions or evaluative statements
    - Maintain a professional, objective tone
    - Present information in a logical, flowing narrative
    - Ensure all key metrics and data points are accurately represented
    - Create clear connections between related pieces of information for investment perspectives.
`);

const founderDynamicsPrompt = PromptTemplate.fromTemplate(`
    Now, using the responses provided above, generate a two-paragraph summary. Focus on capturing the founder’s journey and what brought them here? What motivated them to set up this business. Was there any moment that motivated them to set up this business? What did they do before setting up this venture? How did the founders know each other?

    Company Info:
    {company_data}


       Key points to remember:
    - Focus on facts, avoid opinions or evaluative statements
    - Maintain a professional, objective tone
    - Present information in a logical, flowing narrative
    - Ensure all key metrics and data points are accurately represented
    - Create clear connections between related pieces of information for investment perspectives.
`);

const talkingpointsMarketoppPrompt = PromptTemplate.fromTemplate(`
    Now, using the responses provided above, generate a two-paragraph summary. For the first Paragraph, prepare response in single paragraph by following questions - Do they truly understand the Market Size their business is addressing through a logical response. Do they have a clear unique and defensible business vs their competitors. Do not create a list and prepare a single paragraph - Verify independently the market sizing as well.
    For second paragraph, Does the team have the necessary skill set in the field they are a founder of through previous work experience. If its a technology driven business, is there a CTO (Chief Technical Office) or someone with technical knowledge. Have the team raised money before in this or previous ventures. Do not create a list and prepare a single paragraph. DO they have a proper plan in place? Do they have a plan to execute the vision?

    Company Info:
    {company_data}
`);

const talkingpointsCoachmarketoppPrompt = PromptTemplate.fromTemplate(`
    Now, using the responses provided above, generate a two-paragraph summary.prepare a response in single paragraph by following questions  - Do they have good mentors around them and do they respect them. Have they responded to failure in a positive way where they have shown growth from it and humility. What sacrifices they made to launch this business?. Do not create a list and prepare a single paragraph.

    Company Info:
    {company_data}
`);

const concernsParagraphPrompt = PromptTemplate.fromTemplate(`
    Now, using the responses provided above, generate a two-paragraph summary. Focus on capturing the essence of the business vision, the problem being solved, the founding team's strengths, and their execution strategy. Stick to the facts and do not give opinions.

    Here is the company Info:

    {company_data}.

    Remember to provide output in the following type of  Output Structure. Choose the headings, and paragraphs based on the prompts above.It may or may not be similar to this example below, but follow the format.
    - <b>Product and Market Validation Stage</b>
        Paraph description
    - <b>Limited Financial Runway</b>
        Paraph description
    - <b>Small Team Size</b>
        Paraph description
        - <b>Execution Risks</b>
            Paraph description
            - <b>Market Risks</b>
                Paraph description
                - <b>Product Risks</b>
                    Paraph description
`);

const backgroundInfoPrompt = PromptTemplate.fromTemplate(`

    Summary:

    {Company}, legally known as {legal_name}, was founded by {founding_team} with the purpose of addressing {problem_addressed} in the {industry_sector}. The company, headquartered in {headquarters_location} and incorporated in {incorporation_location}, successfully launched {product_launched} on {launch_date}. Operating in a competitive landscape with players like {competitors}, their unique value proposition, {unique_value_proposition}, sets them apart. The company's go-to-market strategy focuses on reaching {target_customer_location} via {go_to_market_channels}, aiming to capture a significant share of the market.

    Financially, {Company} has reported a revenue of {revenue_last_six_months} and EBITDA of {ebitda_last_six_months} over the last six months, supported by a current cash balance of {cash_balance}. Their monthly burn rate of {monthly_burn_rate} indicates disciplined spending. The founding team’s experience is further highlighted by {team_wins} and their ability to secure prior funding ({prior_funding_experience}). They seek additional funding to meet their goal of raising {fundraising_amount}, with a company valuation of {company_valuation}. Their execution strategy, {execution_vision_team}, reflects their commitment to growth and scalability.
    `);

const scoringPrompt = PromptTemplate.fromTemplate(`

    You have access to some 'BackGround Information' and some 'table data', which you have to use to complete the given task. Analyse the answers and create the scoring system based on the below 'table_data' for this entrepreneur. You have to create the scoring based on the answers/ transcripts and the criteria given in the documents

    Once you complete the scoring, you will need to average each topic to 5. Some topics have multiple aspects being evaluated. For example, Founder dynamics has two aspects being evaluated. You will need to average them together out of 5.

    If information is not available, say N/A.


    Here is the Background Information:

    """
    Summary:

    {Company}, legally known as {legal_name}, was founded by {founding_team} with the purpose of addressing {problem_addressed} in the {industry_sector}. The company, headquartered in {headquarters_location} and incorporated in {incorporation_location}, successfully launched {product_launched} on {launch_date}. Operating in a competitive landscape with players like {competitors}, their unique value proposition, {unique_value_proposition}, sets them apart. The company's go-to-market strategy focuses on reaching {target_customer_location} via {go_to_market_channels}, aiming to capture a significant share of the market.

    Financially, {Company} has reported a revenue of {revenue_last_six_months} and EBITDA of {ebitda_last_six_months} over the last six months, supported by a current cash balance of {cash_balance}. Their monthly burn rate of {monthly_burn_rate} indicates disciplined spending. The founding team’s experience is further highlighted by {team_wins} and their ability to secure prior funding ({prior_funding_experience}). They seek additional funding to meet their goal of raising {fundraising_amount}, with a company valuation of {company_valuation}. Their execution strategy, {execution_vision_team}, reflects their commitment to growth and scalability.

    """
    """


    Here is the table conditions fields that you need to consider for scoring:

    {table_data}

    Think carefully for a long time  and analyze the summary information of the company carefully, so that you can analyze the number of years it has been in business with, the founder dynamics, mentor support, revenue dynamics, ability to hire talent, commercial saviness, purpose, etc and other impportant information that can be helpful for a investor analyzing a startup for investment. Then consider the fields, against which you need to analyze this company for scoring as per the rules and template defined to you. Try your best to come up with a score instead of N/A.

    You will return expected output in JSON format for the updatted table..
     `);

const dominantTraitsPrompt = PromptTemplate.fromTemplate(`
		You are an expert in psychometric analysis and investor psychology, with years of experience in startups analysis. You've been given a JSON data array containing many traits, each with a trait number, name, and a specific description relevant to performance and decision-making. You also have a company summary that includes its vision, mission, target market, and competitive advantages,etc.

		Ypur task is to analyze the company/startup first critically and then evaluate and analyze the traits in the context of this company's profile. Identify the 3 traits that are likely to resonate most with potential investors, given based entirely on the company's profile. Consider the following factors:

	JSON Array of Traits (including trait number, name, and description): {traits_list}
	###

	Company Info:

	{company_data}
	###

	Analyze carefully and inspect the traits that would be most appealing to investors based on the company's profile. For example, ypu can Consider the following criterias:

		Alignment with Company Vision and Mission: Which traits reflect qualities that directly support the company's long-term objectives, making it more attractive to investors with a growth-oriented mindset? (Use context from the given company summary ANalysis)

		Market Positioning: Which traits strengthen the company's ability to differentiate itself in the market, highlighting unique attributes or decision-making skills that enhance investor confidence? (Use context from the given company summary ANalysis)

		Risk and Stability: Which traits suggest an ability to manage risk effectively, adapt to changes, and make intuitive yet reliable decisions under uncertainty?  (Use context from the given company summary ANalysis)

	Please return the TraitNum, names and reason of the top 3 traits you identify, the reason must be a brief explanation of why each trait would be advantageous to this particular company's investor appeal, in  JSON output format.MAke sure to follow the format of the output below.

	Example Output:

			"TraitNum": "16",
			"TraitName": "Trait Name 1",
			"reason": "This trait aligns with the company's vision by..."

	`);

const statusPrompt = PromptTemplate.fromTemplate(`
		You are tasked with evaluating an startup investment against a dynamic investment profile schema, using both the provided 'investment profile questions' and the summary of the 'company data' context. Your goal is to classify the application based on specific eligibility criteria as either "Accepted," "Rejected," or "Potential."

		To do this, you will need to consider:
	1. The investment profile questions
	2. The summary of the company's data and context

	### Criteria for Classification

	1. **Important Questions**:
	   - These questions are marked with an "important" field set to 1.
	   - **Mandatory Matching**: If any important question does not match the applicant's answer exactly, the application should be classified as "Rejected."
	   - **Proceed to Next Step**: If all important questions match, proceed to evaluate the non-important questions.

	2. **Non-Important Questions with Fallbacks (Amber)**:
	   - These questions have an "important" field set to 0 and may include a fallback or "amber" question, indicated by an "amber" field set to 1 and a valid "amber_question_id."
	   - **Matching Logic for Non-Important Questions**:
		 - If a non-important question does not match the applicant’s answer but has an associated amber question, proceed to check the amber question.
		 - **Amber Question Matching**:
			 - If the amber question matches while the original non-important question does not, classify the application as "Potential."
			 - If neither the non-important question nor the amber question match, classify the application as "Rejected."
		 - **No Amber Fallback**: If a non-important question does not match and lacks an amber fallback, classify the application as "Rejected."

	3. **Final Classification**:
	   - **Accepted**: If all important questions match, and either all non-important questions match or are compensated by matching amber questions, classify as "Accepted."
	   - **Rejected**: If any important question fails to match, or if any non-important question without an amber fallback fails to match, classify as "Rejected."
	   - **Potential**: If any non-important questions are mismatched but compensated by matching amber questions, classify as "Potential."

	Now let's review the investment profile questions and company summary, and apply these criteria to classify the application:


	### Investment Profile Questions and Applicant's Answers
	{investmentProfileQuestions}

	### Company Info and Context
	{company_data}
	##

	Okay, now Classify the application based on the above criteria, considering the company’s context. Your final output should be one of the following: "Accepted," "Rejected," or "Potential,", along with a reason for the reasoning based on these criteria.
	The output must be in JSON format,containing these fields shown below.
	status: "Accepted" | "Rejected" | "Potential",
	reason: "Brief explanation of the classification decision."
	`);

export default {
	async queue(event, env) {
		const queryMessage = event.messages[0];
		const queueType = queryMessage.body.queueType;
		const data = queryMessage.body;

		try {
			switch (queueType) {
				case 'generateFounderSummary':
					await generateFounderSummary(data, env);
					break;
				case 'generateFounderDynamics':
					await generateFounderDynamics(data, env);
					break;
				case 'generateScore':
					await generateScore(data, env);
					break;
				case 'generateTalkingPointsMarketOppurtunity':
					await generateTalkingPointsMarketOppurtunity(data, env);
					break;
				case 'generateConcernsParagraph':
					await generateConcernsParagraph(data, env);
					break;
				case 'generateTalkingPointsCoachMarketOppurtunity':
					await generateTalkingPointsCoachMarketOppurtunity(data, env);
					break;
				default:
					console.error(`Unhandled queue: ${event.queue.name}`);
			}
		} catch (error) {
			console.error(`Error processing job in queue ${event.queue.name}:`, error);
		}
	},

	async fetch(request, env) {
		if (request.method === 'OPTIONS') {
			// Handle CORS preflight request
			return new Response(null, {
				status: 204,
				headers: {
					'Access-Control-Allow-Origin': '*', // Or specify specific origin
					'Access-Control-Allow-Methods': 'POST, OPTIONS',
					'Access-Control-Allow-Headers': 'Content-Type',
				},
			});
		}

		// Process POST requests
		if (request.method === 'POST') {
			const requestPayload = await request.json();
			console.log('requestPayload fetch: ===>', requestPayload);
			await env.openAIQueueBinding.send({ ...requestPayload });

			// Add CORS headers to the response
			return new Response('Job added to queue', {
				status: 200,
				headers: {
					'Access-Control-Allow-Origin': '*', // Or specify specific origin
					'Access-Control-Allow-Methods': 'POST, OPTIONS',
					'Access-Control-Allow-Headers': 'Content-Type',
				},
			});
		}

		// Default response for unsupported methods
		return new Response('Method not allowed', { status: 405 });
	},
};

const generateSummaryData = (submission) => {
	return {
		Company: submission['company'],
		legal_name: submission["What's your company's legal name?"],
		description: submission['Please articulate what your business does and sell your vision in no more than 5 sentences.'],
		website: '',
		product_launched: submission['service_launched'],
		launch_date: submission['launch_date'],
		industry_sector: submission['sector'],
		headquarters_location: submission['Where is your company physically headquartered?'],
		incorporation_location: submission['Where is your company incorporated?'],
		revenue_last_six_months: submission['revenue'],
		ebitda_last_six_months: submission['EBITDA'],
		revenue_trend_explanation: submission['If there are any trends in the revenue numbers, please explain.'],
		total_customers_six_months_ago: parseInt(submission['How many total customers did your company have six months ago?']),
		notable_customers: submission[' If any customers or partners are notable, please share.'],
		monthly_burn_rate: submission['burnrate'],
		cash_balance: submission['cash_balance'],
		target_customer_location: submission['target_customers'],
		go_to_market_channels: submission['What are your go-to-market channels?'],
		problem_addressed: submission['What problem does your company address?'],
		competitors: submission['Who are your competitors or substitutes?'],
		unique_value_proposition: submission['What makes your company stand out? (Ensure minimum of 5 sentences)'],
		outside_funding_raised: submission[' Have you already raised any outside funding?'],
		assistance_needed: submission['What functions do you need assistance with in the accelerator program?'],
		applied_to_500_previously: submission['Have you applied to 500 previously?'],
		progress_since_last_application: submission[" What is your company's progress since you last applied?"],
		founding_team: submission['The founding team is the heartbeat of the organisation - please tell us more about them.'],
		reason_for_starting_company: submission['Why did you decide to start this company? (Ensure minimum of 5 sentences)'],
		execution_vision_team: submission['Could you tell us why this founding team will execute its vision? (Ensure minimum of 5 sentences)'],
		prior_funding_experience: submission['Have you or your team raised funding before across other ventures, and if so, how much?'],
		team_wins: submission['Showcase your team’s wins outside of this venture.'],
		business_highlights: [submission['List max 3 business highlights you are most proud of.']],
		full_time_employees: 5,
		part_time_employees: 5,
		co_founders: [''],
		co_founders_linkedin: [''],
		founder_story: submission['List all co-founders with LinkedIn profiles.'],
		expectations_from_investor:
			submission['What would you like to see from your investor? How would you expect your investor to be involved? '] ?? '',
		primary_contact_first_name: submission['First name of primary contact'],
		primary_contact_last_name: submission['Last name of primary contact'],
		primary_contact_email: submission['email'],
		primary_contact_phone: submission['phone'],
		pitch_deck_link: '',
		product_demo_video: '',
		fundraising_amount: submission['fundraise'],
		company_valuation: submission['valuation'],
		equity_split: submission['What is the equity split amongst the founders?'],
		funding_commitments: submission['What are the funding committments?'],
		business_stage: submission['stage'],
	};
};

export const generateConditions = (tableDataList) => {
	let conditionsStr = '';

	for (const row of tableDataList.tables) {
		const points = [row.onePoint, row.twoPoint, row.threePoint, row.fourPoint, row.fivePoint];
		if (row.topic && row.assessment) {
			let conditionBlock = `## ${row.topic} - ${row.assessment.trim()} ##\n`;
			points.forEach((point, i) => {
				if (point && point.trim()) {
					conditionBlock += `If the answer matches '${point.trim()}', assign ${i + 1} point(s).\n`;
				}
			});
			conditionsStr += `${conditionBlock}\n`;
		}
	}

	return conditionsStr;
};

const parseTableData = (tableDataList) => {
	const topics = {};

	for (const row of tableDataList.tables) {
		const topic = row.topic;
		const question = row.assessment;
		const points = [row.onePoint, row.twoPoint, row.threePoint, row.fourPoint, row.fivePoint]
			.filter((p) => p !== undefined && p.trim() !== '')
			.map((p) => p.trim());

		const evaluation = {
			question,
			options: points,
			scores: Array.from({ length: points.length }, (_, i) => i + 1),
		};

		if (topics[topic]) {
			topics[topic].push(evaluation);
		} else {
			topics[topic] = [evaluation];
		}
	}

	const dynamicFields = {};
	for (const [topic, criteria] of Object.entries(topics)) {
		const fieldName = topic.replace(/\s+/g, '_');
		dynamicFields[fieldName] = criteria.length > 1 ? [Array, `Criteria for ${topic}`] : [Object, `Criteria for ${topic}`];
	}

	return dynamicFields;
};

const createDynamicModel = (fieldData) => {
	const dynamicModel = {};

	for (const [fieldName, [fieldType, fieldInfo]] of Object.entries(fieldData)) {
		dynamicModel[fieldName] = {
			type: fieldType,
			description: fieldInfo,
		};
	}

	return dynamicModel;
};

const generateFounderSummary = async (data, env) => {
	try {
		const { companyData, submission_id } = data;
		const founderSummaryChain = founderSummaryPrompt.pipe(llm).pipe(new StringOutputParser());
		const founderSummary = await founderSummaryChain.invoke(companyData);

		await saveToD1(env.DB, submission_id, founderSummary, 'queue-one');
	} catch (e) {
		console.log('error founder summary: ', e);
		return null;
	}
};

const generateFounderDynamics = async (data, env) => {
	try {
		const { companyData, submission_id } = data;
		const founderDynamicsChain = founderDynamicsPrompt.pipe(llm).pipe(new StringOutputParser());
		const founderDynamics = await founderDynamicsChain.invoke(companyData);

		await saveToD1(env.DB, submission_id, founderDynamics, 'queue-one');
	} catch (e) {
		console.log('error founder dynamics: ', e);
		return null;
	}
};

export const generateTalkingPointsMarketOppurtunity = async (data, env) => {
	try {
		const { companyData, submission_id } = data;
		const founderTalkingPointsMarketOppurtunityChain = talkingpointsMarketoppPrompt.pipe(llm).pipe(new StringOutputParser());
		const founderTalkingPointsMarketOppurtunity = await founderTalkingPointsMarketOppurtunityChain.invoke(companyData);

		await saveToD1(env.DB, submission_id, founderTalkingPointsMarketOppurtunity, 'queue-one');
	} catch (e) {
		console.log('error talking points market oppurtunity: ', e);
		return null;
	}
};

export const generateTalkingPointsCoachMarketOppurtunity = async (data, env) => {
	try {
		const { companyData, submission_id } = data;
		const founderTalkingPointsCoachMarketOppurtunityChain = talkingpointsCoachmarketoppPrompt.pipe(llm).pipe(new StringOutputParser());
		const founderTalkingPointsCoachMarketOppurtunity = await founderTalkingPointsCoachMarketOppurtunityChain.invoke(companyData);

		await saveToD1(env.DB, submission_id, founderTalkingPointsCoachMarketOppurtunity, 'queue-one');
	} catch (e) {
		console.log('error talking points coach market oppurtunity: ', e);
		return null;
	}
};

export const generateConcernsParagraph = async (data, env) => {
	try {
		const { companyData, submission_id } = data;
		const founderConcernsParagraphChain = concernsParagraphPrompt.pipe(llm).pipe(new StringOutputParser());
		const founderConcernsParagraph = await founderConcernsParagraphChain.invoke(companyData);

		await saveToD1(env.DB, submission_id, founderConcernsParagraph, 'queue-one');
	} catch (e) {
		console.log('error concerns paragraph: ', e);
		return null;
	}
};

export const generateScore = async (data, env) => {
	try {
		const { companyData, tableData, submission_id } = data;
		console.log('tableData: ', tableData);
		const updatedTableData = generateConditions(tableData);
		console.log('updatedTableData: ', updatedTableData);
		const scoringOutput = generateScoringOutput(JSON.parse(updatedTableData));
		console.log('scoringOutput: ', scoringOutput);
		const parser = new JsonOutputParser(scoringOutput);

		const scoringChain = scoringPrompt.pipe(llm).pipe(parser);
		const scoring = await scoringChain.invoke(companyData, tableData);

		await saveToD1(env.DB, submission_id, scoring, 'queue-one');
	} catch (e) {
		console.log('error generate score: ', e);
		return null;
	}
};

// export const generateDominantTraits = async (data, env) => {
// 	const dominantTraitsChain = await dominantTraitsPrompt.pipe(llm).pipe(new StringOutputParser())
// 	const dominantTraitOutput = dominantTraitsChain.invoke({company_data: company_data,traits_list: transformedDomainatData});
// }

export const generateStatus = async (data, env) => {
	try {
		const { investmentProfileQuestions, submission_id } = data;
		const statusChain = statusPrompt.pipe(llm).pipe(new StringOutputParser());
		const statusOutput = await statusChain.invoke({ company_data: company_data, investmentProfileQuestions: investmentProfileQuestions });

		await saveToD1(env.DB, submission_id, statusOutput, 'queue-one');
	} catch (e) {
		console.log('error generate status: ', e);
		return null;
	}
};

export const generateAllAtOnce = async (data, env) => {
	const founderSummary = await generateFounderSummary(data, env);
	const founderDynamics = await generateFounderDynamics(data, env);
	const scoring = await generateScore(data, env);
	const talkingMarketOpportunity = await generateTalkingPointsMarketOppurtunity(data, env);
	const concerns = await generateConcernsParagraph(data, env);
	const talkingCoach = await generateTalkingPointsCoachMarketOppurtunity(data, env);
	const status = await generateStatus(data, env);

	await saveSubmissionSummaryToD1(env.DB, data.submission_id, data.importRowId, {
		founderSummary,
		founderDynamics,
		talkingMarketOpportunity,
		talkingCoach,
		concerns,
		scoring,
		status,
	});
};

async function saveSubmissionSummaryToD1(db, submission_id, importRowId, data) {
	const query = `UPDATE submissions SET founderSummary = ?, founderDynamics = ?, talkingMarketOpportunity = ?, talkingCoach = ?, concerns = ?, scoring = ?, status = ? WHERE submission_id = ? AND import_row_id = ?;`;
	await db
		.prepare(query)
		.bind(
			data.founderSummary,
			data.founderDynamics,
			data.talkingMarketOpportunity,
			data.talkingCoach,
			data.concerns,
			data.scoring,
			data.status,
			submission_id,
			importRowId
		)
		.run();

	console.log(`Stored submission summary for ${submission_id}`);
}

async function saveToD1(db, submission_id, responseContent, queueName) {
	console.log('id ==>', submission_id, responseContent);
	const query = `INSERT INTO queue_responses (queue_name, original_content, response_content, created_at) VALUES (?, ?, ?, ?);`;
	await db.prepare(query).bind(queueName, originalContent, responseContent, new Date().toISOString()).run();

	console.log(`Stored response for ${queueName}`);
}

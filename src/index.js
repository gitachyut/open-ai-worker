import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser, JsonOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";

const llm = new ChatOpenAI({
	model: "gpt-4o-mini",
	openAIApiKey: ""
});

const founderSummaryPrompt = PromptTemplate.fromTemplate(`
    Now, using the responses provided above, generate a two-paragraph summary. Focus on capturing the essence of the business vision, the problem being solved, the founding team's strengths, and their execution strategy. Stick to the facts and do not give opinions.

    Summary:

    {Company}, legally known as {legal_name}, was founded by {founding_team} with the purpose of addressing {problem_addressed} in the {industry_sector}. The company, headquartered in {headquarters_location} and incorporated in {incorporation_location}, successfully launched {product_launched} on {launch_date}. Operating in a competitive landscape with players like {competitors}, their unique value proposition, {unique_value_proposition}, sets them apart. The company's go-to-market strategy focuses on reaching {target_customer_location} via {go_to_market_channels}, aiming to capture a significant share of the market.

    Financially, {Company} has reported a revenue of {revenue_last_six_months} and EBITDA of {ebitda_last_six_months} over the last six months, supported by a current cash balance of {cash_balance}. Their monthly burn rate of {monthly_burn_rate} indicates disciplined spending. The founding teams experience is further highlighted by {team_wins} and their ability to secure prior funding ({prior_funding_experience}). They seek additional funding to meet their goal of raising {fundraising_amount}, with a company valuation of {company_valuation}. Their execution strategy, {execution_vision_team}, reflects their commitment to growth and scalability.
`);

const founderDynamicsPrompt = PromptTemplate.fromTemplate(`
    Now, using the responses provided above, generate a two-paragraph summary. Focus on capturing the essence of the business vision, the problem being solved, the founding team's strengths, and their execution strategy. Stick to the facts and do not give opinions.

    Summary:

    {Company}, legally registered as {legal_name}, is focused on addressing {problem_addressed} within the {industry_sector}. The founders, {founding_team}, launched {product_launched} on {launch_date}, aiming to capture {target_customer_location}. They’ve built the company from {headquarters_location}, leveraging a unique value proposition of {unique_value_proposition} to stand out from competitors like {competitors}. The founders, driven by {reason_for_starting_company}, are strategically targeting customers through channels like {go_to_market_channels}.

    Financial performance in the last six months shows a revenue of {revenue_last_six_months} and EBITDA of {ebitda_last_six_months}, with {total_customers_six_months_ago} customers as of six months ago, including notable clients such as {notable_customers}. With a monthly burn rate of {monthly_burn_rate} and a current cash balance of {cash_balance}, the team is efficiently managing its resources. Their experience with {prior_funding_experience} has enabled them to raise {outside_funding_raised}, positioning them well for future fundraising goals of {fundraising_amount}. Their vision and execution strategy are supported by the team’s achievements like {team_wins} and a clear focus on {execution_vision_team}.
`);

const talkingpointsMarketoppPrompt = PromptTemplate.fromTemplate(`
    Now, using the responses provided above, generate a two-paragraph summary. Focus on capturing the essence of the business vision, the problem being solved, the founding team's strengths, and their execution strategy. Stick to the facts and do not give opinions.

    Summary:

    {Company}, legally known as {legal_name}, has been making strides in the {industry_sector} by solving {problem_addressed} with their innovative product/service, {product_launched}. Since launching on {launch_date}, the company has targeted {target_customer_location} through {go_to_market_channels}. Their financial performance, with a revenue of {revenue_last_six_months} and EBITDA of {ebitda_last_six_months}, shows steady growth. They differentiate themselves through {unique_value_proposition}, while key competitors include {competitors}.

    The company has a strong financial position with a cash balance of {cash_balance}, though their monthly burn rate of {monthly_burn_rate} reflects ongoing operational costs. The founding team’s leadership, including {co_founders}, has propelled the company forward, driving customer acquisition and growth, including {total_customers_six_months_ago} customers and notable partnerships with {notable_customers}. They have also successfully raised {outside_funding_raised}, and are looking to secure additional investment of {fundraising_amount} at a valuation of {company_valuation}, with {equity_split} equity split among founders. Their execution strategy, {execution_vision_team}, ensures they are well-positioned to seize new market opportunities.
`);

const talkingpointsCoachmarketoppPrompt = PromptTemplate.fromTemplate(`
    Now, using the responses provided above, generate a two-paragraph summary. Focus on capturing the essence of the business vision, the problem being solved, the founding team's strengths, and their execution strategy. Stick to the facts and do not give opinions.

    Summary:

    {Company}, formally known as {legal_name}, was established to address {problem_addressed} within the {industry_sector}. The founders, {co_founders}, have successfully launched {product_launched} on {launch_date}, from their headquarters in {headquarters_location}. Their strategy revolves around targeting {target_customer_location} through {go_to_market_channels}, differentiating the company from competitors such as {competitors} with their unique value proposition: {unique_value_proposition}. The founders’ decision to create the company stems from {reason_for_starting_company}, and their execution strategy revolves around {execution_vision_team}.

    The company has made significant progress in its financial performance, generating {revenue_last_six_months} in revenue and {ebitda_last_six_months} in EBITDA over the last six months. With a burn rate of {monthly_burn_rate} and a current cash balance of {cash_balance}, they are managing expenses prudently. Their ability to attract notable customers like {notable_customers} has been instrumental in growth. They have raised {outside_funding_raised} and are currently seeking to raise {fundraising_amount}, at a valuation of {company_valuation}, to fuel their next stage of growth. The founders’ past wins, such as {team_wins}, and prior funding experience ({prior_funding_experience}) further demonstrate their readiness for expansion.
`);

const concernsParagraphPrompt = PromptTemplate.fromTemplate(`
    Now, using the responses provided above, generate a two-paragraph summary. Focus on capturing the essence of the business vision, the problem being solved, the founding team's strengths, and their execution strategy. Stick to the facts and do not give opinions.

    Summary:

    {Company}, legally known as {legal_name}, is focused on addressing {problem_addressed} in the {industry_sector}. The product/service, {product_launched}, launched on {launch_date}, aims to tackle key issues in the market. The company's go-to-market strategy targets {target_customer_location}, leveraging {go_to_market_channels}, but competitors such as {competitors} could present challenges. Despite these challenges, their unique value proposition—{unique_value_proposition}—offers a strong differentiator.

    Financially, while the company reported {revenue_last_six_months} in revenue and {ebitda_last_six_months} EBITDA over the last six months, there are concerns over the monthly burn rate of {monthly_burn_rate} and the sustainability of their cash balance ({cash_balance}). Their goal of raising {fundraising_amount} at a valuation of {company_valuation} will be crucial to maintaining growth. With {full_time_employees} full-time and {part_time_employees} part-time employees, the team’s ability to execute effectively will be vital. The company's previous wins ({team_wins}) and the leadership of the founding team {founding_team} offer hope, but challenges in competition and resource management remain key concerns.

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



export default {
	async queue(event, env) {
		const queryMessage = event.messages[0]
		const queueType = queryMessage.body.queueType
		const data = queryMessage.body
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
		const requestPayload = await request.json();
		await env.openAIQueueBinding.send({ ...requestPayload });
		return new Response("Job added to queue", { status: 200 });
	},
};

const generateFounderSummary = async (data, env) => {
	try {
		const { companyData, submission_id } = data;
		const founderSummaryChain = founderSummaryPrompt.pipe(llm).pipe(new StringOutputParser());
		const founderSummary = await founderSummaryChain.invoke(companyData);

		await saveToD1(env.DB, submission_id, founderSummary, "queue-one");
	} catch (e) {
		console.log("error founder summary: ", e)
		return null
	}
}

const generateFounderDynamics = async (data, env) => {
	try {
		const { companyData, submission_id } = data;
		const founderDynamicsChain = founderDynamicsPrompt.pipe(llm).pipe(new StringOutputParser());
		const founderDynamics = await founderDynamicsChain.invoke(companyData)

		await saveToD1(env.DB, submission_id, founderDynamics, "queue-one");
	} catch (e) {
		console.log("error founder dynamics: ", e)
		return null
	}
}

export const generateTalkingPointsMarketOppurtunity = async (data, env) => {
	try {
		const { companyData, submission_id } = data;
		const founderTalkingPointsMarketOppurtunityChain = talkingpointsMarketoppPrompt.pipe(llm).pipe(new StringOutputParser());
		const founderTalkingPointsMarketOppurtunity = await founderTalkingPointsMarketOppurtunityChain.invoke(companyData)

		await saveToD1(env.DB, submission_id, founderTalkingPointsMarketOppurtunity, "queue-one");
	} catch (e) {
		console.log("error talking points market oppurtunity: ", e)
		return null
	}
}

export const generateTalkingPointsCoachMarketOppurtunity = async (data, env) => {
	try {
		const { companyData, submission_id } = data;
		const founderTalkingPointsCoachMarketOppurtunityChain = talkingpointsCoachmarketoppPrompt.pipe(llm).pipe(new StringOutputParser());
		const founderTalkingPointsCoachMarketOppurtunity = await founderTalkingPointsCoachMarketOppurtunityChain.invoke(companyData)

		await saveToD1(env.DB, submission_id, founderTalkingPointsCoachMarketOppurtunity, "queue-one");
	} catch (e) {
		console.log("error talking points coach market oppurtunity: ", e)
		return null
	}
}

export const generateConcernsParagraph = async (data, env) => {
	try {
		const { companyData, submission_id } = data;
		const founderConcernsParagraphChain = concernsParagraphPrompt.pipe(llm).pipe(new StringOutputParser());
		const founderConcernsParagraph = await founderConcernsParagraphChain.invoke(companyData)

		await saveToD1(env.DB, submission_id, founderConcernsParagraph, "queue-one");
	} catch (e) {
		console.log("error concerns paragraph: ", e)
		return null
	}
}

export const generateScore = async (data, env) => {
	try {
		const { companyData, tableData, submission_id } = data;
		console.log("tableData: ", tableData)
		const updatedTableData = generateConditions(tableData)
		console.log("updatedTableData: ", updatedTableData)
		const scoringOutput = generateScoringOutput(JSON.parse(updatedTableData))
		console.log("scoringOutput: ", scoringOutput)
		const parser = new JsonOutputParser(scoringOutput)

		const scoringChain = scoringPrompt.pipe(llm).pipe(parser);
		const scoring = await scoringChain.invoke(companyData, tableData);

		await saveToD1(env.DB, submission_id, scoring, "queue-one");
	} catch (e) {
		console.log("error generate score: ", e)
		return null
	}
}


async function saveToD1(db, submission_id, responseContent, queueName) {
	console.log("id ==>", submission_id, responseContent);
	//   const query = `INSERT INTO queue_responses (queue_name, original_content, response_content, created_at) VALUES (?, ?, ?, ?);`;
	//   await db.prepare(query)
	//     .bind(queueName, originalContent, responseContent, new Date().toISOString())
	//     .run();

	console.log(`Stored response for ${queueName}`);
}


